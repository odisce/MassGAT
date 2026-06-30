import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import os


class SinglePeakDataset(Dataset):
    def __init__(self, root_dir, cache_file=None, load_from_cache=True, augmentation="none", aug_params=None):
        """
        Args:
            root_dir: Folder containing .npy files
            cache_file: Path to cache file (optional)
            load_from_cache: If True, load from cache if available
        """
        self.root_dir = root_dir
        self.augmentation = augmentation
        self.aug_params = aug_params if aug_params is not None else {}
        if cache_file is None:
            cache_file = os.path.join(root_dir, 'single_peak_dataset_cache.pt')
        self.cache_file = cache_file
        self.data = []
        self.labels = []
        if load_from_cache and os.path.exists(self.cache_file):
            cache = torch.load(self.cache_file)
            self.data = cache['data']
            self.labels = cache['labels']
            print(f"Loaded SinglePeakDataset from cache: {self.cache_file} ({len(self.data)} samples)")
        else:
            print(f"Building SinglePeakDataset from raw files in: {root_dir}")
            self.files = list(Path(root_dir).rglob("peak_*.npy"))
            for path in self.files:
                raw_arr = np.load(path).astype(np.float32)
                if raw_arr.shape != (72,):
                    print(f"[WARNING] File {path} has shape {raw_arr.shape} (expected (72,)). Values: {raw_arr}")
                raw_x = torch.from_numpy(raw_arr)
                sc_length = int(raw_x[7].item())
                if sc_length <= 3:
                    continue  # Skip peaks with scan length <= 3
                fname = path.name
                if "true" in fname:
                    y = 1.0
                else:
                    y = 0.0
                # --- FEATURE ENGINEERING (MATCH loader.py) ---
                mz = raw_x[0]
                into = raw_x[2]
                rt = raw_x[1]
                rtmin = raw_x[3]
                imax = raw_x[5]
                rt_seq = raw_x[8:40]
                int_seq = raw_x[40:72]
                log_into = torch.log1p(into)
                log_imax = torch.log1p(imax)
                mz_int = torch.floor(mz)
                mz_float = mz - mz_int
                mz_norm = mz_int / 1000
                rt_norm = rt / torch.max(rt, torch.tensor(1.0))
                rt_relative_seq = rt_seq - rtmin
                rt_relative_seq[rt_seq == 0] = 0
                int_max = int_seq.max() if int_seq.numel() > 0 else 1.0
                int_relative_seq = int_seq / (int_max + 1e-6)
                x = raw_x.clone()
                x[0] = mz_norm
                x[1] = rt_norm
                x[2] = log_into
                x[5] = log_imax
                x[6] = mz_float
                x[8:40] = rt_relative_seq
                x[40:72] = int_relative_seq
                self.data.append(x)
                self.labels.append(torch.tensor([y], dtype=torch.float))
            torch.save({'data': self.data, 'labels': self.labels}, self.cache_file)
            print(f"Saved SinglePeakDataset to cache: {self.cache_file} ({len(self.data)} samples)")
            print(f"Dataset built with {len(self.data)} samples, including {sum(l.item() for l in self.labels)} positives.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx], self.labels[idx]
        import torch
        import random
        x_aug = x.clone().unsqueeze(0)
        aug_type = self.augmentation
        params = self.aug_params
        # Set default augmentation parameters 
        intensity_noise_scale = params.get("intensity_noise_scale", 0.1)
        rt_shift_scale = params.get("rt_shift_scale", 0.5)
        mass_jitter_scale = params.get("mass_jitter_scale", 5e-6)
        profile_noise_std = params.get("profile_noise_std", 0.02)
        gaussian_noise_std = params.get("gaussian_noise_std", 0.01)
        feature_dropout_prob = params.get("feature_dropout_prob", 0.1)
        aug_prob = params.get("aug_prob", 0.7)  # probability to apply each augmentation

        if self.augmentation != "none":
            # Intensity scaling
            if aug_type in ["intensity", "mixed"] and random.random() < aug_prob:
                scale_factor = 1.0 + (torch.rand(1) * 2 - 1) * intensity_noise_scale
                log_scale = torch.log(scale_factor)
                x_aug[:,2] += log_scale  # log_into
                x_aug[:,5] += log_scale  # log_imax
                x_aug[:,40:72] *= scale_factor
            # RT shift
            if aug_type in ["rt_shift", "mixed"] and random.random() < aug_prob:
                node_shift = (torch.rand(1) * 2 - 1) * rt_shift_scale
                x_aug[:,1] += node_shift  # rt_norm
            # Mass jitter
            if aug_type in ["mass_jitter", "mixed"] and random.random() < aug_prob:
                jitter = torch.randn(1) * mass_jitter_scale
                x_aug[:,6] += jitter  # mz_float
            # Profile noise
            if aug_type in ["profile_noise", "mixed"] and random.random() < aug_prob:
                noise = torch.randn(1,32) * profile_noise_std
                mask = (x_aug[:,40:72] > 0)
                x_aug[:,40:72] += noise * mask
            # Gaussian noise
            if aug_type in ["gaussian", "mixed"] and random.random() < aug_prob:
                x_aug += torch.randn_like(x_aug) * gaussian_noise_std
            # Feature dropout
            if aug_type in ["dropout", "mixed"] and random.random() < aug_prob:
                dropout_mask = torch.rand(1,32) > feature_dropout_prob
                x_aug[:,40:72] *= dropout_mask.float()
        x = x_aug.squeeze(0)
        return x, y


def extract_peaks_from_pt(pt_path):
    import torch
    data_dict = torch.load(pt_path, map_location='cpu')
    node_features = data_dict['node_features']  # shape: (num_nodes, 72)
    roi_values = data_dict['roi_values']        # shape: (num_nodes,)
    num_nodes = node_features.shape[0]
    peaks = []
    for i in range(num_nodes):
        raw_x = node_features[i].clone()
        mz = raw_x[0]
        rt = raw_x[1]
        into = raw_x[2]
        rtmin = raw_x[3]
        rtmax = raw_x[4]
        imax = raw_x[5]
        rt_seq = raw_x[8:40]
        int_seq = raw_x[40:72]

        log_into = torch.log1p(into)
        log_imax = torch.log1p(imax)
        mz_int = torch.floor(mz)
        mz_float = mz - mz_int
        mz_norm = mz_int / 1000.0
        rt_norm = rt / torch.max(rt, torch.tensor(1.0))

        # RT normalization
        rt_relative_seq = rt_seq - rtmin
        rt_relative_seq[rt_seq == 0] = 0

        # Intensity normalization by max
        int_max = int_seq.max() if int_seq.numel() > 0 else 1.0
        int_relative_seq = int_seq / (int_max + 1e-6)

        max_rt = torch.max(rtmax, torch.tensor(1.0))
        rtmin = rtmin / max_rt
        rtmax = rtmax / max_rt
        rt_norm = rt_norm / max_rt
        rt_relative_seq = rt_relative_seq / max_rt


        x = raw_x.clone()
        x[0] = mz_norm
        x[1] = rt_norm
        x[2] = 0.0 
        x[5] = 0.0  
        x[6] = mz_float
        x[8:40] = rt_relative_seq
        x[40:72] = int_relative_seq

        peaks.append((x, int(roi_values[i].item())))
    return peaks