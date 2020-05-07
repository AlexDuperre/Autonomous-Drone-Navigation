import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import h5py
import cv2
from matplotlib.pylab import cm
import random

from util.MidAirSegmenter import DNDSegmenter

class DND(Dataset):
    def __init__(self, data_dir, transform = None, frames_nb=30, subsegment_nb=1, overlap=20, goal_chg=False):
        self.frame_nb = frames_nb
        data_segmenter = DNDSegmenter(data_dir)
        self.Table = data_segmenter.segment((self.frame_nb,), overlap,subsegment_nb)
        random.shuffle(self.Table)
        self.Table = self.Table[0:int(len(self.Table)*1)]
        self.goal_chg = goal_chg
        self.transform = transform

    def __getitem__(self, idx):
        # initialize inputs:
        depth = np.zeros((self.frame_nb,96,160))
        rel_orientation = np.zeros((self.frame_nb))
        rel_goalx = np.zeros((self.frame_nb))
        rel_goaly = np.zeros((self.frame_nb))
        goal_orientation = np.zeros((self.frame_nb))
        GT = np.ones((self.frame_nb))
        mask = np.zeros((self.frame_nb))

        segment = self.Table[idx][0]
        path = self.Table[idx][1]
        f = h5py.File(path, "r")
        length = len(f["GT"][segment[1]:segment[1] + segment[0]])
        depth[0:length, :,:] = f["depth"][segment[1]:segment[1]+segment[0]]
        rel_orientation[0:length] = f["rel_orientation"][segment[1]:segment[1]+segment[0]]
        rel_goalx[0:length] = f["rel_goalx"][segment[1]:segment[1]+segment[0]]
        rel_goaly[0:length] = f["rel_goaly"][segment[1]:segment[1]+segment[0]]
        goal_orientation [0:length] = f["goal_orientation"][segment[1]:segment[1] + segment[0]]
        GT[0:length] = f["GT"][segment[1]:segment[1] + segment[0]]
        mask[0:length] = 1

        # test frames
        Testframes = False
        if Testframes:
            for i in range(len(GT)):
                image = depth[i]
                rel_goal = rel_goalx[i], rel_goaly[i]
                image = display_trajectory(image, rel_goal, fix_angle(np.pi*rel_orientation[i]))
                cv2.imshow("Depth", image)
                cv2.waitKey(200)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if (np.random.random()>0.4) & self.goal_chg:
            rel_orientation, rel_goalx, rel_goaly = destination_swap(rel_orientation, rel_goalx, rel_goaly, goal_orientation)



        # test frames
        if Testframes:
            for i in range(len(GT)):
                image = depth[i]
                rel_goal = rel_goalx[i], rel_goaly[i]
                image = display_trajectory(image, rel_goal, np.pi*rel_orientation[i])
                cv2.imshow("Depth", image)
                cv2.waitKey(200)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if self.transform:
            image = self.transform(depth)

        return (depth, rel_orientation, rel_goalx, rel_goaly, GT.astype(int)-1, length, mask)

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
        if goal_orientation[i] < 0:
            goal_orientation[i] = goal_orientation[i] + 2 * np.pi


        # Calculate current orientation
        current_orientation = -np.pi*rel_orientation[i] + goal_orientation[i]
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