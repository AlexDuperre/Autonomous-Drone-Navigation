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

class class_counter():
    def __init__(self):
        self.f = open("class_count.txt","r+")
        self.data = self.f.read()

        if not self.data:
            self.data = [0,0,0,0,0,0]

        else:
            self.data = self.data.split(",")
            self.data = [eval(nb) for nb in self.data]

    
    def add(self, count):
        self.data = list(np.asarray(self.data) + count)
        
    def save_data(self):
        self.f.truncate(0)
        self.f.seek(0)
        self.f.write(str(self.data).strip("[]"))
        self.f.close()



def replace_keys_parallel(GT):

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
    # counter.add(GT)

    # print(bytes(str(GT),encoding='utf8'))
    return bytes(str(GT),encoding='utf8')


def rescale_video(image):
    #rescale and normalize video
    image = cv2.resize(image, dsize=(160,92), interpolation=cv2.INTER_CUBIC)
    image = (image - image.min())/(image.max() - image.min() + 0.000001)
    return image

def rescale_depth(depth):
    depth = cv2.resize(depth, dsize=(320, 192), interpolation=cv2.INTER_CUBIC)
    return depth

def normalize_depth(depth):
    depth = (depth - depth.min())/(6.0 - depth.min())
    return depth


def fix_angle(rel_orientation):
    # makes sure the angle is between pi and - pi
    if rel_orientation > np.pi:
        rel_orientation = -1.0*(2*np.pi - rel_orientation)
    elif rel_orientation < -np.pi:
        rel_orientation = 2*np.pi - (-1.0*rel_orientation)
    # normalises the angle between -1 an 1
    return np.around(np.float32(rel_orientation/ np.pi), 3)


def fun(idx, h5pyData):
    data = {}

    # Replace GT keys for class number
    GT = h5pyData["GT"][idx]
    data["GT"] = replace_keys_parallel( GT)

    # rescale & normalize video
    image = h5pyData["video"][idx][0]
    data["video"] = rescale_video(image)

    # Normalize depth
    depth = h5pyData["depth"][idx]
    data["depth"] = rescale_depth(depth)

    # fix and normalize the rel_orientation angle
    rel_orientation = h5pyData["rel_orientation"][idx]
    data["rel_orientation"] = fix_angle(rel_orientation)

    return data




def main(file_path):
    #"./path_7_5_001.h5"

    counter = class_counter()
    with h5py.File(file_path, "a") as f:

        # Get length of dataset
        length = len(f["GT"])
        print("Length = ", length)

        # Save dataset into a new list variable
        data = {}
        data["GT"] = f["GT"][0:length]
        # data["video"] = f["video"][0:length]
        data["depth"] = f["depth"][0:length]
        # data["rel_orientation"] = f["rel_orientation"][0:length]

        # Create a new pool
        p = Pool(8)

        # Set max chunk size and max i to avoid indexing error
        max_items = 500
        max_i = length//max_items
        for i in range(max_i+1):
            d = {}

            # Check if we need to match the index to the end of the file then use
            # multiprocessing to clean the dataset chunks and extend the list
            if i==max_i:
                d["GT"] = data["GT"][max_items * i:]
                # d["video"] = data["video"][max_items * i:]
                d["depth"] = data["depth"][max_items * i:]
                # d["rel_orientation"] = data["rel_orientation"][max_items * i:]
                partial_func = partial(fun, h5pyData=d)
                if i==0:
                    out = p.map(partial_func, range(length-max_items*max_i))
                else:
                    out.extend(p.map(partial_func, range(length-max_items*max_i)))
            else:
                d["GT"] = data["GT"][max_items * i: max_items*(i+1)]
                # d["video"] = data["video"][max_items * i: max_items*(i+1)]
                d["depth"] = data["depth"][max_items * i: max_items*(i+1)]
                # d["rel_orientation"] = data["rel_orientation"][max_items * i: max_items*(i+1)]
                partial_func = partial(fun, h5pyData=d)
                if i==0:
                    out = p.map(partial_func, range(max_items))
                else:
                    out.extend(p.map(partial_func, range(max_items)))

        # Aggregate data as ndarray to their specific key
        data = {}
        for k in out[0].keys():
            data[k] = np.stack(list(data[k] for data in out))

        # Calculate the number of classes in this trajectory
        onehot = np.zeros((len(data["GT"]),6))
        onehot[np.arange(len(data["GT"])), np.int8(data["GT"])] = 1

        # save number of items per categories up till now
        count = np.sum(onehot, axis=0)
        counter.add(count)
        counter.save_data()

        # Resize raw video
        f["video"].resize((length,1,192,320,3))

        # Resize data array to (length,1,192,320,3)
        data["video"] = np.expand_dims(data["video"], axis=1)

        # Resize depth
        f["depth"].resize((length, 192, 320))



        # Rewrite new data
        f["GT"][:] = data["GT"]
        f["video"][:] = data["video"]
        f["depth"][:] = data["depth"]
        f["rel_orientation"][:] = data["rel_orientation"]



if __name__ == '__main__':
    main(sys.argv[1])
