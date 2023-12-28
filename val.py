import argparse
from torch.utils.tensorboard import SummaryWriter
from model import *
from data.loader import data_loader
from utils import *
import time


parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")
parser.add_argument("--dataset_name", default="eth", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=4, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)
parser.add_argument("--skip", default=1, type=int)
parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--best_k", default=20, type=int)
parser.add_argument("--print_every", default=10, type=int)
parser.add_argument("--use_gpu", default=1, type=int)
parser.add_argument("--gpu_num", default="0", type=str)
parser.add_argument("--d_model", default=256, type=int)
parser.add_argument("--n_heads", default=4, type=int)
parser.add_argument("--n_layers", default=1, type=int)
parser.add_argument("--noise_dim", default=64, type=int)
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    logging.info("Initializing test dataset")
    _, test_loader = data_loader(args, phase='test')

    writer = SummaryWriter()

    model = MRGTraj(args)
    print(model)
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint["state_dict"])

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    model.cuda()
    model.eval()

    ade, fde = evaluate(args, test_loader, model, args.best_k)
    print(
        "Dataset: {}, Pred Len: {}, ADE: {:.12f}, FDE: {:.12f}".format(
            args.dataset_name, args.pred_len, ade, fde
        )
    )


def evaluate_helper(error):
    sum_ = 0
    error = torch.stack(error, dim=1)
    _error, index = torch.min(error, dim=1)
    _error = torch.sum(_error, dim=0)
    sum_ += _error
    return sum_, index


def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    batch_count, times_count = 0, 0

    with torch.no_grad():
        for batch in loader:
            t0 = time.time()
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

            ade, fde = [], []
            total_traj += traj_abs.size(1)
            pred_traj_fakes = []
            past_traj = torch.cat((traj_abs[:args.obs_len], traj_rel[:args.obs_len]), dim=-1)
            for _ in range(num_samples):
                pred_traj_fake_rel = generator.inference(
                    past_traj.transpose(0, 1), batch_mask)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, traj_abs[args.obs_len - 1])

                ade.append(displacement_error(
                    pred_traj_fake, traj_abs[args.obs_len:], mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], traj_abs[-1], mode='raw'
                ))
                pred_traj_fakes.append(pred_traj_fake)
            ade_sum, index = evaluate_helper(ade)
            fde_sum, _ = evaluate_helper(fde)
            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
            batch_count += len(seq_start_end)
            t1 = time.time()
            times_count += (t1 - t0)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde


if __name__ == "__main__":
    args = parser.parse_args()
    args.base_dir = "checkpoints/"
    args.resume = args.base_dir + args.dataset_name + "_best_checkpoint.pth.tar"
    torch.manual_seed(72)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)
