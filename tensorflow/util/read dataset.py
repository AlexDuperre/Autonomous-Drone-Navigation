import cv2
import h5py
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import cm

def fix_angle(rel_orientation):
    if rel_orientation > np.pi:
        rel_orientation = -1.0*(2*np.pi - rel_orientation)
    elif rel_orientation < -np.pi:
        rel_orientation = 2*np.pi - (-1.0*rel_orientation)

    return np.around(np.float32(rel_orientation), 3)

def display_trajectory(image, rel_goal, rel_orientation):

    display_angle = rel_orientation + (np.pi/2)
    arrowpoint = (int(320 + 40 * np.cos(display_angle)), int(320 - 40 * np.sin(display_angle)))
    rel_orientation = fix_angle(rel_orientation)
    med = np.median(image)
    # image = ((image - image.min()) * (1 / (6 - 0) * 255)).astype('uint8') # to be used with raw dataset
    image = image*2.0  # to be used with the normalized dataset (adjust scale)
    image = np.uint8(cm.jet(image) * 255)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    image = cv2.resize(image, (int(640), int(360)))
    image = cv2.arrowedLine(image, (320,320), arrowpoint, (0,0,255), 2, cv2.LINE_AA)
    # text = "%.3f"%(min) + "m." + "                        " + "%.3f"%(max) + "m."
    image = cv2.putText(image, "%.3f"%(np.sqrt(rel_goal[0]**2 + rel_goal[1]**2))+'m.', (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    image = cv2.putText(image, str(rel_orientation)+" rad", (0,320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    image = cv2.putText(image, str(med), (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    return image

def main():
    filepath = "/tmp/DATASET/dataset/path_2_9_000.h5"
    f = h5py.File(filepath, "r")

    for idx in range(len(f["depth"])):
        image = f["depth"][idx]
        rel_goal = [f["rel_goalx"][idx], f["rel_goaly"][idx]]
        rel_orientation = f["rel_orientation"][idx]
        image = display_trajectory(image, rel_goal, rel_orientation)
        print(idx)
        cv2.imshow("Depth", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    f.close()

if __name__ == '__main__':
    main()