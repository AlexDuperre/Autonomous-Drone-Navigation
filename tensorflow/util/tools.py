import os
from os import listdir
from os.path import isfile, join
import string
import cv2, queue, threading
from matplotlib.pylab import cm
import numpy as np
import pyexcel_ods
import matplotlib.pyplot as plt
import torch

"""
bufferless VideoCapture functions
"""
class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

"""
Transform the input image to add a normalizing factor and navigation elements (distances and navigation arrow)
"""
def post_treatment(image, rel_goal, arrowpoint, min, max):
    image = ((image - image.min()) * (1 / (6 - 0) * 255)).astype('uint8')
    image = np.uint8(cm.jet(image)*255)
    image = cv2.cvtColor(image,cv2.COLOR_RGBA2BGR)
    image = cv2.resize(image, (int(640), int(360)))
    image = cv2.arrowedLine(image, (320,320), arrowpoint, (0,0,255), 2, cv2.LINE_AA)
    text = "%.3f"%(min) + "m." + "                        " + "%.3f"%(max) + "m."
    image = cv2.putText(image, "%.3f"%(np.sqrt(rel_goal[0]**2 + rel_goal[1]**2))+'m.', (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    image = cv2.putText(image, text, (0,320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    return image



"""
Transformation of raw ROS telemetry
"""
def telemetry_transform(goal, split_data):
    try:
        # calculate relative position
        rel_goal = (goal[0] - float(split_data[0]), goal[1] - float(split_data[1]))
    except:
        rel_goal = (0,0)

    # calculate absolute angle (-pi,pi) to objective
    goal_orientation = np.arctan2(rel_goal[1], rel_goal[0])
    # get current orientation in ROS environment
    current_orientation = float(split_data[3])
    # put all angles between 0 and 2pi
    if goal_orientation < 0:
        goal_orientation = goal_orientation + 2 * np.pi
    if current_orientation < 0:
        current_orientation = current_orientation + 2 * np.pi
    # Calculate relative angle to objective (*-1 to have the arrow on the right when negative.)
    rel_orientation = -1 * (current_orientation - goal_orientation)
    display_angle = rel_orientation + (np.pi / 2)
    return rel_orientation, display_angle, rel_goal, goal_orientation



"""
Extract destinations from excel spreadsheet
"""
def get_goals(path, world_number):
    data = pyexcel_ods.get_data(path)
    # print(data)
    return data["Sheet1"][world_number][1:]


"""
Extract column data from list
"""
def extract_data(data,idx):
    return [item[idx] for item in data]


"""
Iterate a filename from filename_000.xx to fielname_001.hxx
"""
def indexer(file_path):
    iteration = 1
    file_exist = True
    while file_exist:
        if os.path.isfile(file_path):
            file_path = file_path[:-6] + (str(iteration)).zfill(3) + ".h5"
            iteration += 1
        else:
            file_exist = False


    return file_path

"""
Compute path data for the Display_path function 
"""
def compute_paths(rel_orientation, relx, rely, predictions, path, dtheta=0.15):
    destination = get_goals("./destinations.ods", 3)
    start = eval(destination[eval(path.split("_")[-3])])
    theta = np.tan(rely[0]/relx[0]) + rel_orientation[0] * np.pi
    pred_points_x = [start[0]]
    pred_points_y = [start[1]]
    dx = 0
    dy = 0
    for i, pred in enumerate(predictions):
        if pred == 0:
            dx = np.cos(theta)
            dy = np.sin(theta)
        if pred == 1:
            theta = theta + dtheta
        if pred == 2:
            theta = theta - dtheta
        if pred == 3:
            theta = theta + dtheta
            dx = np.cos(theta)
            dy = np.sin(theta)
        if pred == 4:
            theta = theta - dtheta
            dx = np.cos(theta)
            dy = np.sin(theta)

        pred_points_x.append(pred_points_x[i] + dx)
        pred_points_y.append(pred_points_y[i] + dy)
        dx = 0
        dy = 0
    return [pred_points_x,pred_points_y]


"""
Display predicted path for the validation run compared for the ground truth 
"""
def display_paths(sets, epoch, nb):
    plt.plot(sets[0][0],sets[0][1])
    plt.plot(sets[1][0], sets[1][1])
    plt.legend(["pred","true"])
    root = "./val paths/" + "epoch " + str(epoch) + "/"
    if not os.path.exists(root):
        os.makedirs(root)
    plt.savefig(root + str(nb) + ".jpg")
    plt.show(block=False)
    plt.close()


def dynamic_distribution(labels):
    """
    Creates a probability distribution of the targets by taking a sample of N/2 before timestep t and N/2 after.
    For timesteps < N/2, the distribution will be made of the N-t elements after and the t elements before.
    Idem for the N/2 timsteps at the end of the sequence.
    """
    batch_size, seq_len = labels.shape
    dist = torch.zeros([batch_size,seq_len,5])
    N_2 = 3
    for i in range(seq_len):
        if i < N_2:
            dist[:, i, :] = calculate_stats(labels[:, 0 : 2*N_2+1])
        elif i + N_2 > seq_len :
            dist[:, i, :] = calculate_stats(labels[:, -2 * N_2-2 : -1])
        else:
            dist[:,i,:] = calculate_stats(labels[:,i-N_2:i+N_2])
    return dist

def calculate_stats(label_seq):
    stats = torch.stack([(label_seq == 0).sum(1), (label_seq == 1).sum(1), (label_seq == 2).sum(1), (label_seq == 3).sum(1),
                 (label_seq == 4).sum(1)], 1)
    return stats/stats[0].sum().float()


def main():
    labels = torch.randint(0,4,[100,33])
    print(labels)
    distrib = dynamic_distribution(labels)
    print(distrib)
    print(distrib.shape)

if __name__ == '__main__':
    main()
