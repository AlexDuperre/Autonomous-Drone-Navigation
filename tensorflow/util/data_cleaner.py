import h5py
from itertools import compress
import cv2
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import cm

class class_counter():
    def __init__(self):
        self.f = open("class_count.txt","r+")
        self.data = self.f.read()

        if not self.data:
            self.data = [0,0,0,0,0,0]

        else:
            self.data = self.data.split(",")
            self.data = [eval(nb) for nb in self.data]

        # f.write(str(self.data))
        # f.close()
    
    def add(self, class_nb):
        self.data[class_nb] += 1
        
    def save_data(self):
        self.f.truncate(0)
        self.f.seek(0)
        self.f.write(str(self.data).strip("[]"))
        self.f.close()

        

def replace_keys(f, counter, idx, GT):

    GT = GT.decode()
    # print(GT)

    items = GT.split("+")

    # Clear all other keys
    possible_keys = ["q", "w", "e", "None"]

    res = [word in possible_keys for word in items]

    items = list(compress(items, res))


    # Si q et e en meme temps
    # retirer les deux car les deux s'opposeent: aucun mouvement sauf si w est dans la combinaison
    possible_keys = ["q", "e"]

    # finds if q or e are present
    res = [word in possible_keys for word in items]

    if sum(res)==2:
        # Flip boolean values to pop them out
        res = [not i for i in res]

        # Pop the q and e
        items = list(compress(items, res))


    # Insert None if list is empty
    if not items:
        items = ["None"]


    # class 0 : None
    possible_keys = ["None"]

    res = [word in possible_keys for word in items]

    if all(res):
        GT = 0

    # class 1  : w
    possible_keys = ["w"]

    res = [word in possible_keys for word in items]

    if all(res):
        GT = 1

    # class 2  : q
    possible_keys = ["q"]

    res = [word in possible_keys for word in items]

    if all(res):
        GT = 2

    # class 3  : e
    possible_keys = ["e"]

    res = [word in possible_keys for word in items]

    if all(res):
        GT = 3

    # class 4  : w+q
    possible_keys = ["w", "q"]

    res = [word in possible_keys for word in items]

    if all(res) & (len(res) > 1):
        GT = 4

    # class 5  : w+e
    possible_keys = ["w", "e"]

    res = [word in possible_keys for word in items]

    if all(res) & (len(res) > 1):
        GT = 5

    # Add one to the class counter
    counter.add(GT)

    # print(bytes(str(GT),encoding='utf8'))
    f["GT"][idx] = bytes(str(GT),encoding='utf8')



def rescale_video(f, idx, image):
    #rescale and normalize video
    image = cv2.resize(image, dsize=(320,192), interpolation=cv2.INTER_CUBIC)
    image = (image - image.min())/(image.max() - image.min())
    f["video"][idx, 0, 0:192, 0:320] = image


def normalize_depth(f, idx, depth):
    depth = (depth - depth.min())/(6.0 - depth.min())
    f["depth"][idx] = depth


def fix_angle(f, idx, rel_orientation):
    # makes sure the angle is between pi and - pi
    if rel_orientation > np.pi:
        rel_orientation = -1.0*(2*np.pi - rel_orientation)
    elif rel_orientation < -np.pi:
        rel_orientation = 2*np.pi - (-1.0*rel_orientation)
    # normalises the angle between -1 an 1
    f["rel_orientation"][idx] = np.around(np.float32(rel_orientation/ np.pi), 3)



def main():
    file_path = "/tmp/dataset/path_1_21_000.h5"#""./path_7_5_001.h5"

    counter = class_counter()
    f = h5py.File(file_path, "a")

    for idx in range(len(f["GT"])):
        print("iteration # ", idx)

        # Replace GT keys for class number
        GT = f["GT"][idx]
        replace_keys(f, counter, idx, GT)

        # rescale & normalize video
        image = f["video"][idx][0]
        rescale_video(f, idx, image)

        # Normalize depth
        depth = f["depth"][idx]
        normalize_depth(f, idx, depth)

        # fix and normalize the rel_orientation angle
        rel_orientation = f["rel_orientation"][idx]
        fix_angle(f, idx, rel_orientation)

    # Resize raw video
    f["video"].resize((381,1,192,320,3))

    # save number of items per cathegoies up till now
    counter.save_data()

    # close hdf5 file
    f.close()

if __name__ == '__main__':
    main()
