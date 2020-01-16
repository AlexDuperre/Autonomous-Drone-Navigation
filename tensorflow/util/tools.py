import cv2, queue, threading
from matplotlib.pylab import cm
import numpy as np

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

def telemetry_transform(goal, split_data):
    try:
        rel_goal = (goal[0] - float(split_data[0]), goal[1] - float(split_data[1]))
    except:
        rel_goal = (0,0)
    goal_orientation = np.arctan2(rel_goal[1], rel_goal[0])
    current_orientation = float(split_data[3])
    if goal_orientation < 0:
        goal_orientation = goal_orientation + 2 * np.pi
    if current_orientation < 0:
        current_orientation = current_orientation + 2 * np.pi
    rel_orientation = -1 * (current_orientation - goal_orientation)
    display_angle = rel_orientation + (np.pi / 2)
    return rel_orientation, display_angle, rel_goal, rel_orientation