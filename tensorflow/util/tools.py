import os
from os import listdir
from os.path import isfile, join
import string
import cv2, queue, threading
from matplotlib.pylab import cm
import numpy as np
import pyexcel_ods


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
    # Calculate relative angle to objective (*-1 to have the arrow on the right when negative)
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


def main():
    # goals = get_goals("../destinations.ods",1)
    #
    # goal = eval(goals[0])
    # print(goal[0])
    files = indexer("../dataset/ardrone_office/path_1_1_000.h5")
    print(files)

if __name__ == '__main__':
    main()
