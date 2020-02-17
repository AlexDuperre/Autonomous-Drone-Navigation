import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import h5py

from util.MidAirSegmenter import MidAirDataSegmenter

class Dataset(Dataset):
    def __init__(self, data_dir, transform = None, min_frames=30, max_frames=50, overlap=20):
        data_segmenter = MidAirDataSegmenter(data_dir)
        self.Table = data_segmenter.segment((min_frames,), overlap)

        self.transform = transform

    def __getitem__(self, idx):
        segment = self.Table[idx][0]
        path = self.Table[idx][1]
        f = h5py.File(path, "r")
        depth = f["depth"][segment[1]:segment[1]+segment[0]]
        rel_orientation = f["rel_orientation"][segment[1]:segment[1]+segment[0]]
        rel_goalx = f["rel_goalx"][segment[1]:segment[1]+segment[0]]
        rel_goaly = f["rel_goaly"][segment[1]:segment[1]+segment[0]]
        # if self.transform:
        #     image = self.transform(image)
        return (depth, rel_orientation,rel_goalx, rel_goaly)

    def __len__(self):
        return len(self.Table)