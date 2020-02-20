import os
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
import math
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torchvision import transforms

from util.Exceptions import InvalidSizeException


class TrajectorySegmenter:
    def segment_trajectory(self, overlap, subsequence_frame_nb, subsequence_nb, trajectory,
                           trajectory_length):
        sequence_length_range = (subsequence_frame_nb[0] * subsequence_nb,)

        if sequence_length_range is None:
            start_frames = [0]
            sequence_lengths = trajectory_length
            sequence = {"sequence_length": sequence_lengths, "start_frame_index": start_frames}
            dataframe = pd.DataFrame(sequence)
            # serialization_destination.create_dataset('trajectory_segments', data=dataframe.to_numpy())
            return dataframe.values.tolist()

        elif len(sequence_length_range) == 1:
            if sequence_length_range[0]>trajectory_length:
                start_frames = [0]
                sequence_lengths = trajectory_length//subsequence_frame_nb[0] * subsequence_frame_nb[0]
            else:
                start_frames = list(
                    range(0, trajectory_length - sequence_length_range[0], sequence_length_range[0] - overlap))
                if trajectory_length - sequence_length_range[0] == 0:
                    start_frames = [0]

                dropped_frames = (trajectory_length - 1) - (start_frames[-1] + sequence_length_range[0])

                sequence_lengths = [sequence_length_range[0]] * len(start_frames)

                if dropped_frames//subsequence_frame_nb[0] > 0:
                    start_frames.append(start_frames[-1]+sequence_length_range[0])
                    sequence_lengths.append(dropped_frames//subsequence_frame_nb[0] * subsequence_frame_nb[0])
                    dropped_frames -= dropped_frames//subsequence_frame_nb[0] * subsequence_frame_nb[0]

                if dropped_frames > 0:
                    print("Last {} frames not used for trajectory {}".format(dropped_frames,
                                                                             trajectory))


            sequence = {"sequence_length": sequence_lengths, "start_frame_index": start_frames}
            dataframe = pd.DataFrame(sequence)
            # serialization_destination.create_dataset('trajectory_segments', data=dataframe.to_numpy())
            return dataframe.values.tolist()

        elif len(sequence_length_range) == 2:
            sequence = {"sequence_length": [], "start_frame_index": []}
            start = 0
            while True:
                sequence_length = np.random.randint(sequence_length_range[0], sequence_length_range[1] + 1)
                if start + sequence_length < trajectory_length:
                    sequence["sequence_length"].append(sequence_length)
                    sequence["start_frame_index"].append(start)
                    start += (sequence_length - overlap)
                else:
                    if trajectory_length - start >= sequence_length_range[0]:
                        sequence["sequence_length"].append(trajectory_length - start)
                        sequence["start_frame_index"].append(start)
                    else:
                        print("Last {} frames not used for trajectory {}".format(trajectory_length - start,
                                                                                 trajectory))
                    break
            dataframe = pd.DataFrame(sequence)
            # serialization_destination.create_dataset('trajectory_segments', data=dataframe.to_numpy())
            return dataframe.values.tolist()
        else:
            raise InvalidSizeException("sequence_length_range should have length of either 1 or 2")


class DNDSegmenter(TrajectorySegmenter):
    def __init__(self, path_to_data: str):
        self.path_to_data = path_to_data

    def segment(self, subsequence_frame_nb: Tuple = None, overlap: int = 0, subsequence_nb: int = 0):
        """Segments the trajectories found into, sequences of length sequence_length_range.
        If no arguments are passed or if sequence_length_range is None (default) then
        the whole trajectory will be used a a single segment.
        """
        segment_list = []
        path_list = []
        for root, dirs, files in os.walk(self.path_to_data):

            for name in files:
                if name.endswith(".h5"):
                    path = os.path.join(root,name).replace('\\','/')
                    print(path)
                    datasets = h5py.File(path, "r+")
                    trajectory_length = self.__get_trajectory_length__(datasets)
                    segments = self.segment_trajectory(overlap, subsequence_frame_nb, subsequence_nb,
                                                       name, trajectory_length)
                    segment_list.extend(segments)
                    path_list.extend(len(segments)*[path])
        return list(zip(segment_list,path_list))

    def __get_trajectory_length__(self, datasets):
            return datasets["GT"].len()