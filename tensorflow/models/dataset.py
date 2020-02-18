import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import h5py
import cv2
from matplotlib.pylab import cm

from util.MidAirSegmenter import DNDSegmenter

class DND(Dataset):
    def __init__(self, data_dir, transform = None, frames_nb=30, overlap=20):
        data_segmenter = DNDSegmenter(data_dir)
        self.Table = data_segmenter.segment((frames_nb,), overlap)

        self.transform = transform

    def __getitem__(self, idx):
        segment = self.Table[idx][0]
        path = self.Table[idx][1]
        f = h5py.File(path, "r")
        depth = f["depth"][segment[1]:segment[1]+segment[0]]
        rel_orientation = f["rel_orientation"][segment[1]:segment[1]+segment[0]]
        rel_goalx = f["rel_goalx"][segment[1]:segment[1]+segment[0]]
        rel_goaly = f["rel_goaly"][segment[1]:segment[1]+segment[0]]
        goal_orientation = f["goal_orientation"][segment[1]:segment[1] + segment[0]]
        GT = f["GT"][segment[1]:segment[1] + segment[0]]

        # test frames
        Testframes = False
        if Testframes:
            for i in range(len(GT)):
                image = depth[i]
                rel_goal = rel_goalx[i], rel_goaly[i]
                image = display_trajectory(image, rel_goal, fix_angle(rel_orientation[i]))
                cv2.imshow("Depth", image)
                cv2.waitKey(1000)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if False:
            rel_orientation, rel_goalx, rel_goaly = destination_swap(rel_orientation, rel_goalx, rel_goaly, goal_orientation)


        # test frames
        if Testframes:
            for i in range(len(GT)):
                image = depth[i]
                rel_goal = rel_goalx[i], rel_goaly[i]
                image = display_trajectory(image, rel_goal, rel_orientation[i])
                cv2.imshow("Depth", image)
                cv2.waitKey(1000)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # if self.transform:
        #     image = self.transform(image)

        return (depth, rel_orientation, rel_goalx, rel_goaly, GT.astype(int))

    def __len__(self):
        return len(self.Table)


def destination_swap(rel_orientation, rel_goalx, rel_goaly, goal_orientation):
    new_rel_orientation = np.zeros(len(rel_orientation))
    for i in range(len(rel_orientation)):
        # Subtracts the final coordinates of the segments to define the end of the segment as the goal
        rel_goalx[i] -= rel_goalx[-1]
        rel_goaly[i] -= rel_goaly[-1]

        # Calculate new goal orientation
        new_goal_orientation = np.arctan2(rel_goaly[i], rel_goalx[i])
        if new_goal_orientation < 0:
            new_goal_orientation = new_goal_orientation + 2 * np.pi

        # makes sure the angle is between pi and - pi
        if goal_orientation[i] > np.pi:
            goal_orientation[i] = -1.0 * (2 * np.pi - goal_orientation[i])
        # Calculate current orientation
        current_orientation = np.pi*rel_orientation[i] + goal_orientation[i]
        if current_orientation < 0:
            current_orientation = current_orientation + 2 * np.pi

        # Calculate new rel_orientation
        new_rel_orientation[i] = -1 * (current_orientation - new_goal_orientation)
        new_rel_orientation[i] = fix_angle(new_rel_orientation[i])
    return new_rel_orientation, rel_goalx, rel_goaly

def fix_angle(rel_orientation):
    # makes sure the angle is between pi and - pi
    if rel_orientation > np.pi:
        rel_orientation = -1.0*(2*np.pi - rel_orientation)
    elif rel_orientation < -np.pi:
        rel_orientation = 2*np.pi - (-1.0*rel_orientation)
    # normalises the angle between -1 an 1
    return np.around(rel_orientation/ np.pi, 3)

def display_trajectory(image, rel_goal, rel_orientation):

    display_angle = rel_orientation + (np.pi/2)
    arrowpoint = (int(320 + 40 * np.cos(display_angle)), int(320 - 40 * np.sin(display_angle)))


    # image = ((image - image.min()) * (1 / (6 - 0) * 255)).astype('uint8') # to be used with raw dataset
    image = image*6.0  # to be used with the normalized dataset (adjust scale)
    image = np.uint8(cm.jet(image)*255.0)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    image = cv2.resize(image, (int(640), int(360)))
    image = cv2.arrowedLine(image, (320,320), arrowpoint, (0,0,255), 2, cv2.LINE_AA)
    # text = "%.3f"%(min) + "m." + "                        " + "%.3f"%(max) + "m."
    image = cv2.putText(image, "%.3f"%(np.sqrt(rel_goal[0]**2 + rel_goal[1]**2))+'m.', (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    image = cv2.putText(image, str(rel_orientation)+" rad", (0,320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    return image