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

This program will compute the distance of each trajectories, generate an image of the trajectory and create a file wih the mean distance for each trajectories  as title

"""

"""
Select the number associated to the world

1 - Luxury home
2-  Luxury home 2nd floor
3 - Bar
4 - Machine room
5 - Mechanical plant
6 - office
7 - Resto bar
8 - Scanned home
9 - Scanned home 2nd floor
"""



def main():
    world_no = 6
    world_strings = ["luxury_home", "luxury_home_2e_floor", "bar", "machine_room", "mechanical_plant", "office",
                     "resto_bar", "scanned_home", "scanned_home_2e_floor"]
    world_ref_coord = [[0, 0], [0, 0], [10.5, 1], [0, 0], [20.5, 2.5], [1, 31], [0, 0], [0, 0], [0, 0]]
    world_ref_dims = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 22, 0, 11], [0, 0, 0, 0], [0, 25, 0, 23], [0, 30, 0, 49],
                      [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    destination_list = get_goals("../destinations.ods", world_no)

    path_to_data = "/windows/aldupd/light dataset V2/ardrone_" + world_strings[world_no-1] +"/"


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

                            base = "../Test_paths/ardrone_" + world_strings[world_no-1] + "/GT/path_" + str(prev_dest_id+1) + "_" + str(dest_id+1)
                            if not os.path.exists(base):
                                os.makedirs(base)

                            background = plt.imread("../Test_paths/ardrone_" + world_strings[world_no-1] + "_raw.png")

                            plt.imshow(background, extent=world_ref_dims[world_no-1])

                            plt.plot(world_ref_coord[world_no-1][0] - destination[1] + y, world_ref_coord[world_no-1][1] + destination[0] - x, "bo-")

                            plt.axis(world_ref_dims[world_no-1])

                            plt.savefig(base + "/dist_" + str(np.around(dist,3)) + "_.png", dpi=450, transparent=True)
                            plt.clf()

    for root, dirs, files in os.walk("../Test_paths/ardrone_" + world_strings[world_no-1] + "/GT/"):

                for dir in dirs:
                    path = os.path.join(root,dir).replace('\\','/')
                    mean = 0
                    i = 0
                    for file in os.listdir(path):
                        if file.endswith(".png"):
                            mean += np.float(file.split("_")[-2])
                            i += 1
                    mean = mean/(i+1e-10)
                    f = open(path + "/mean_dist_" + str(np.around(mean,3)) + ".txt", "w+")
                    f.close()





if __name__ == '__main__':
    main()