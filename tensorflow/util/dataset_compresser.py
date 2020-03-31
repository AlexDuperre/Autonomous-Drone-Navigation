import sys
import h5py
import os
from itertools import compress
import cv2
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import cm

from collections import defaultdict
from multiprocessing import Pool
from functools import partial
from functools import reduce



def main(file_path):

    new_path = "/media/aldupd/UNTITLED 2/light dataset"

    world_no = 9
    world_strings = ["luxury_home", "luxury_home_2e_floor", "bar", "machine_room", "mechanical_plant", "office",
                     "resto_bar", "scanned_home", "scanned_home_2e_floor"]

    for j in range(world_no):
        # Create saving directory if it doesn't exist
        directory = new_path + "/ardrone_" + world_strings[j - 1]
        if not os.path.exists(directory):
            os.makedirs(directory)

    # file_path = "./path_7_5_001.h5"
    print("Working on : ", file_path)

    with h5py.File(file_path, "a") as f:
        depth = f["depth"][:]
        rel_orientation = f["rel_orientation"][:]
        rel_goalx = f["rel_goalx"][:]
        rel_goaly = f["rel_goaly"][:]
        goal_orientation = f["goal_orientation"][:]
        GT = f["GT"][:]

    title = new_path + "/" + file_path.split("/")[-2] + "/" + file_path.split("/")[-1]
    print("copied to ; ",title)
    with h5py.File(title, "w") as hdf:
        hdf.create_dataset("depth", data=depth, compression="gzip")
        hdf.create_dataset("rel_goalx", data=rel_goalx, compression="gzip")
        hdf.create_dataset("rel_goaly", data=rel_goaly, compression="gzip")
        hdf.create_dataset("goal_orientation", data=goal_orientation, compression="gzip")
        hdf.create_dataset("rel_orientation", data=rel_orientation, compression="gzip")
        hdf.create_dataset("GT", data=GT, compression="gzip")


if __name__ == '__main__':
    main(sys.argv[1])
