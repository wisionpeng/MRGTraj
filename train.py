import argparse
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import *
from data.loader import data_loader
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")

parser.add_argument("--dataset_name", default="eth", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=0, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)
parser.add_argument("--skip", default=1, type=int)

parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--num_epochs", default=300, type=int)

parser.add_argument("--best_k", default=20, type=int)
parser.add_argument("--print_every", default=20, type=int)
parser.add_argument("--use_gpu", default=1, type=int)
parser.add_argument("--gpu_num", default="0", type=str)

parser.add_argument("--d_model", default=256, type=int)
parser.add_argument("--n_heads", default=4, type=int)
parser.add_argument("--n_layers", default=1, type=int)
parser.add_argument("--noise_dim", default=64, type=int)
parser.add_argument("--lr", default=1e-3)
parser.add_argument('--start_test', default=10, type=int)
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")


best_ade, best_fde, best_epoch = 100, 100, 0


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    logging.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, phase='train')
    logging.info("Initializing val dataset")
    _, val_loader = data_loader(args, phase='val')

    writer = SummaryWriter()

    model = MRGTraj(args)

    model.cuda()
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    global best_ade, best_fde, best_epoch
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("Restoring from checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.num_epochs):
        train(args, model, train_loader, optimizer, epoch, writer)
        if epoch > args.start_test:
            ade, fde = validate(args, model, val_loader, epoch, writer)
            is_best = ade <= best_ade or fde <= best_fde
            best_ade = min(ade, best_ade)
            best_fde = min(fde, best_fde)
            if is_best:
                best_epoch = epoch

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_ade": best_ade,
                    "best_fde": best_fde,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                args.checkpoint_dir + f"/best_checkpoint.pth.tar",
            )
            logging.info(
                " ***** Best_ADE  {best_ade:.3f} Best_FDE  {best_fde:.3f} in Epoch  {best_epoch}"
                    .format(best_ade=best_ade, best_fde=best_fde, best_epoch=best_epoch)
            )
    writer.close()


def train(args, model, train_loader, optimizer, epoch, writer):
    losses = AverageMeter("Loss", ":.6f")
    progress = ProgressMeter(
        len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch)
    )
    model.train()
    sample_nums = 0
    for batch_idx, batch in enumerate(train_loader):
        sample_num = batch[-1]
        batch = [tensor.cuda() for tensor in batch[:-1]]
        (traj_abs,
            traj_rel,
            batch_mask,
            non_linear_ped,
            loss_mask,
            seq_start_end,
            frames
        ) = batch
        optimizer.zero_grad()
        loss = torch.zeros(1).to(traj_abs)
        past_traj = torch.cat((traj_abs[:args.obs_len], traj_rel[:args.obs_len]), dim=-1)
        future_traj = torch.cat((traj_abs[args.obs_len:], traj_rel[args.obs_len:]), dim=-1)  # for MRGTraj_sl
        pred_traj_fake_rel, mu, log_var = model(
            past_traj.transpose(0, 1), future_traj.transpose(0, 1), batch_mask)  # for KL(p(z|...), N(0, 1))

        l2_loss_sum_rel = l2_loss(
            pred_traj_fake_rel, traj_rel[-args.pred_len:], loss_mask=loss_mask[-args.pred_len:])
        kld_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # for KL(p(z|...), N(0, 1))

        loss += (l2_loss_sum_rel + kld_loss.mean())
        losses.update(loss.item(), traj_abs.shape[1])
        loss.backward()
        optimizer.step()
        sample_nums += sample_num
        # print(batch_ped_num, ped_num)
        if batch_idx % args.print_every == 0:
            progress.display(sample_nums)
    writer.add_scalar("train_loss", losses.avg, epoch)


def validate(args, model, val_loader, epoch, writer):
    ade = AverageMeter("ADE", ":.6f")
    fde = AverageMeter("FDE", ":.6f")

    progress = ProgressMeter(len(val_loader), [ade, fde], prefix="Test: ")
    sample_nums = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            sample_num = batch[-1]
            batch = [tensor.cuda() for tensor in batch[:-1]]
            (
                traj_abs,
                traj_rel,
                batch_mask,
                non_linear_ped,
                loss_mask,
                seq_start_end,
                frames
            ) = batch
            pred_traj_fakes = []
            past_traj = torch.cat((traj_abs[:args.obs_len], traj_rel[:args.obs_len]), dim=-1)
            for _ in range(args.best_k):
                pred_traj_fake_rel = model.inference(
                    past_traj.transpose(0, 1), batch_mask)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, traj_abs[args.obs_len-1])
                pred_traj_fakes.append(pred_traj_fake)
            pred_traj_fakes = torch.stack(pred_traj_fakes, dim=0)

            ade_, fde_ = ade_fde_of_samples(pred_traj_fakes, traj_abs[args.obs_len:])

            ade.update(ade_, traj_abs.shape[1])
            fde.update(fde_, traj_abs.shape[1])
            sample_nums += sample_num
            if i % args.print_every == 0:
                progress.display(sample_nums)

        logging.info(
            " * ADE  {ade.avg:.3f} FDE  {fde.avg:.3f}".format(ade=ade, fde=fde)
        )
        writer.add_scalar("val_ade", ade.avg, epoch)
    return ade.avg, fde.avg


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    return ade, fde


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def save_checkpoint(state, is_best, filename="best_checkpoint.pth.tar"):
    if is_best:
        torch.save(state, filename)
        logging.info("-------------- lower ade ----------------")


if __name__ == "__main__":
    args = parser.parse_args()
    checkpoint_dir = "./checkpoints/" + args.dataset_name
    args.checkpoint_dir = checkpoint_dir
    if os.path.exists(checkpoint_dir) is False:
        os.makedirs(checkpoint_dir)
    train_log = args.dataset_name + "_train.log"
    set_logger(os.path.join(args.checkpoint_dir, train_log))
    main(args)
