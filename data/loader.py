from torch.utils.data import DataLoader as DataLoader_test
from data.trajectories import TrajectoryDataset, seq_collate
from data.dataloader import DataLoader

DATASET_NAME_TO_NUM = {
    'eth': 0,
    'hotel': 1,
    'zara1': 2,
    'zara2': 3,
    'univ': 4
}


def data_loader(args, phase="train"):
    data_dirs = ['./dataset/eth/univ', './dataset/eth/hotel',
                 './dataset/ucy/zara/zara01', './dataset/ucy/zara/zara02',
                 './dataset/ucy/univ/students001', './dataset/ucy/univ/students003',
                 './dataset/ucy/univ/uni_examples', './dataset/ucy/zara/zara03']

    skip = [6, 10, 10, 10, 10, 10, 10, 10]

    train_set = [i for i in range(len(data_dirs))]

    assert args.dataset_name in DATASET_NAME_TO_NUM.keys(), 'Unsupported dataset {}'.format(args.dataset_name)

    dataset_id = DATASET_NAME_TO_NUM[args.dataset_name]

    if dataset_id == 4 or dataset_id == 5:
        dataset_id = [5]
    else:
        dataset_id = [dataset_id]

    for x in dataset_id:
        train_set.remove(x)

    if phase == 'train':
        dir = [data_dirs[x] for x in train_set]
    elif phase == 'val':
        dir = [data_dirs[x] for x in dataset_id]
    elif phase == 'test':
        dir = [data_dirs[x] for x in dataset_id]

    dset = TrajectoryDataset(
        dir,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)
    if phase == 'train' or phase == 'val':
        loader = DataLoader(
            dset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.loader_num_workers,
            collate_fn=seq_collate,
            pin_memory=True)
    elif phase == 'test':
        loader = DataLoader_test(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=args.loader_num_workers,
            collate_fn=seq_collate,
            pin_memory=True)
    return dset, loader
