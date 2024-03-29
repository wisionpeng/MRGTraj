import logging
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def seq_collate(data):
    (
        seq_list,
        # pred_seq_list,
        seq_rel_list,
        # pred_seq_rel_list,
        non_linear_ped_list,
        loss_mask_list,
        ped_num_list,
        frames_list
    ) = zip(*data)

    _len = [len(seq) for seq in seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]
    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    traj_abs = torch.cat(seq_list, dim=0).permute(2, 0, 1)
    # pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    traj_rel = torch.cat(seq_rel_list, dim=0).permute(2, 0, 1)
    # pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)

    frames = torch.Tensor(frames_list)
    # print(frames)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    sum_ped_num = len(ped_num_list)
    batch_mask = torch.zeros(traj_abs.shape[1], traj_abs.shape[1])
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        # cur_pos = traj_abs[7, start:end].unsqueeze(0).repeat(end-start, 1, 1)
        # mini_mask = torch.sqrt(torch.sum((cur_pos - cur_pos.transpose(0, 1)) ** 2, dim=-1)) <= 10
        # batch_mask[start:end, start:end] = mini_mask.to(int) - torch.eye(end-start)
        batch_mask[start:end, start:end] = 1
    out = [
        traj_abs,
        # pred_traj,
        traj_rel,
        # pred_traj_rel,
        batch_mask,
        non_linear_ped,
        loss_mask,
        seq_start_end,
        frames,
        sum_ped_num
    ]

    return tuple(out)


def read_file(_path, delim="\t"):
    data = []
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "
    with open(_path, "r") as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    # print(t)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    # print(res_x, res_y)
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
        self,
        data_dir,
        obs_len=8,
        pred_len=12,
        skip=1,
        threshold=0.01,
        min_ped=1,
        delim="\t",
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = self.data_dir
        # all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        seq_frames = []
        for directory in all_files:
            file_path = os.path.join(directory, 'true_pos_.csv')
            data = np.genfromtxt(file_path, delimiter=',')
            # print(data.shape)
            data = data.transpose(1, 0)
            # print(data.shape)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                # curr_seq_data is a 20 length sequence
                curr_seq_data = np.concatenate(
                    frame_data[idx: idx + self.seq_len], axis=0
                )
                # print(curr_seq_data.shape)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []

                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_frame = curr_ped_seq[0, 0]
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq

                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    # rel_curr_ped_seq[:, 0] = rel_curr_ped_seq[:, 1]
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1
                    # curr_frame =

                if num_peds_considered >= min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    seq_frames.append(curr_frame)

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        seq_frames = np.asarray(seq_frames)
        # Convert numpy -> Torch Tensor
        self.traj_abs = torch.from_numpy(seq_list).type(
            torch.float
        )
        # self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len :]).type(
        #     torch.float
        # )
        self.traj_rel = torch.from_numpy(seq_list_rel).type(
            torch.float
        )
        # self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len :]).type(
        #     torch.float
        # )

        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        self.seq_frames = torch.from_numpy(seq_frames)
        # print(self.seq_frames)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.traj_abs[start:end, :],
            # self.pred_traj[start:end, :],
            self.traj_rel[start:end, :],
            # self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
            end - start,
            self.seq_frames[index]

        ]
        return out
