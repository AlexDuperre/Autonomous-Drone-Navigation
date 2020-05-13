import os
import h5py
import torch
from models.dataset import DND
from models.model import LSTMModel
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from util.tools import get_goals

"""

This program will compute the distance of each trajectories and generate an image of the trajectory

"""

def main():
    path_to_data = "/windows/aldupd/val-test set"

    world_no = 3
    destination_list = get_goals("../destinations.ods", world_no)

    for root, dirs, files in os.walk(path_to_data):

                for name in files:
                    if name.endswith(".h5"):
                        path = os.path.join(root,name).replace('\\','/')
                        path_splitted = path.split("_")
                        prev_dest_id = np.int(path_splitted[-3])
                        dest_id = np.int(path_splitted[-2])
                        destination = eval(destination_list[dest_id])

                        with h5py.File(path, "r+") as datasets:

                            # Calculate distance
                            dist = 0
                            for i in range(datasets["rel_goalx"].len() - 1):
                                dist += np.sqrt((datasets["rel_goalx"][i + 1] - datasets["rel_goalx"][i]) ** 2 + (datasets["rel_goaly"][i + 1] - datasets["rel_goaly"][i]) ** 2)

                            # Generate trajectory image
                            x = datasets["rel_goalx"][0:-1:3]
                            y = datasets["rel_goaly"][0:-1:3]

                            base = "../Test_paths/ardrone_bar/GT/" + "path_" + str(prev_dest_id+1) + "_" + str(dest_id+1)
                            if not os.path.exists(base):
                                os.makedirs(base)

                            background = plt.imread("../Test_paths/ardrone_bar_raw.png")

                            plt.imshow(background, extent=[0, 22, 0, 11])

                            plt.plot(10 - destination[1] + y, 1 + destination[0] - x, "bo-")

                            plt.axis([0, 22, 0, 12])

                            plt.savefig(base + "/dist_" + str(np.around(dist,3)) + "_.png", dpi=450, transparent=True)
                            plt.clf()

    for root, dirs, files in os.walk("../Test_paths/ardrone_bar/GT/"):

                for dir in dirs:
                    path = os.path.join(root,dir).replace('\\','/')
                    mean = 0
                    i = 0
                    for file in os.listdir(path):
                        if file.endswith(".png"):
                            mean += np.float(file.split("_")[-2])
                            i += 1
                    mean = mean/(i+1e-10)
                    f = open(path + "/mean_dist_" + str(np.around(mean)) + ".txt", "w+")
                    f.close()





if __name__ == '__main__':
    main()