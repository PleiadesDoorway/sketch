import numpy as np
import torch
from torch.utils.data import Dataset

MAX_LEN = 100

def generate_attention_mask(stroke_length):
    mask = torch.zeros((MAX_LEN, MAX_LEN), dtype=torch.float32)
    mask[stroke_length:, :] = -1e8
    mask[:, stroke_length:] = -1e8
    return mask

def generate_padding_mask(stroke_length):
    mask = torch.ones((MAX_LEN, 1), dtype=torch.float32)
    mask[stroke_length:, :] = 0
    return mask

class Quickdraw414k4VanillaTransformer(Dataset):
    def __init__(self, sketch_list, data_dict):
        with open(sketch_list) as f:
            sketch_lines = f.readlines()
        
        self.coordinate_urls = [line.strip().split(' ')[0].replace('png', 'npy') for line in sketch_lines]
        self.labels = [int(line.strip().split(' ')[-1]) for line in sketch_lines]
        self.data_dict = data_dict

    def __len__(self):
        return len(self.coordinate_urls)

    def __getitem__(self, idx):
        coord, flag_bits, stroke_len = self.data_dict[self.coordinate_urls[idx]]
        label = self.labels[idx]

        # numpy -> tensor
        if isinstance(coord, np.ndarray):
            coord = torch.from_numpy(coord).float()
        else:
            coord = coord.float()

        if isinstance(flag_bits, np.ndarray):
            flag_bits = torch.from_numpy(flag_bits).float()
        else:
            flag_bits = flag_bits.float()

        # 保证 flag_bits shape [100,1]，便于 cat
        if flag_bits.ndim == 1:
            flag_bits = flag_bits.unsqueeze(1)

        # masks
        attention_mask = generate_attention_mask(stroke_len)
        padding_mask = generate_padding_mask(stroke_len)
        position_encoding = torch.arange(MAX_LEN, dtype=torch.float32).unsqueeze(1)

        return (coord, label, flag_bits, stroke_len, attention_mask, padding_mask, position_encoding)
