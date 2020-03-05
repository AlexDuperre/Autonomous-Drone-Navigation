import sys
import h5py
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
    # file_path = "./path_7_5_001.h5"
    print("Working on : ", file_path)
    with h5py.File(file_path, "a") as f:
        depth = f["depth"][:]
        rel_orientation = f["rel_orientation"][:]
        rel_goalx = f["rel_goalx"][:]
        rel_goaly = f["rel_goaly"][:]
        goal_orientation = f["goal_orientation"][:]
        GT = f["GT"][:]

        # get index of what to keep
        indexes = GT.astype(int)!=0

        # keeping all but class 0 entries
        depth = depth[indexes]
        rel_orientation = rel_orientation[indexes]
        rel_goalx = rel_goalx[indexes]
        rel_goaly = rel_goaly[indexes]
        goal_orientation = goal_orientation[indexes]
        GT = GT[indexes]

        # delete datasets
        f.__delitem__("depth")
        f.__delitem__("rel_orientation")
        f.__delitem__("rel_goalx")
        f.__delitem__("rel_goaly")
        f.__delitem__("goal_orientation")
        f.__delitem__("GT")

        # reassing datasets
        f["depth"] = depth
        f["rel_orientation"] =  rel_orientation
        f["rel_goalx"] = rel_goalx
        f["rel_goaly"] = rel_goaly
        f["goal_orientation"] = goal_orientation
        f["GT"] = GT



if __name__ == '__main__':
    main(sys.argv[1])
