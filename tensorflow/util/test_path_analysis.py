import os
import numpy as np

for root, dirs, files in os.walk("../Test_paths/ardrone_bar/73 Full loss - 0.2 goal 0.4 flip - mechanical plant val-test/"):

    for dir in dirs:
        path = os.path.join(root, dir).replace('\\', '/')
        mean = 0
        i = 0
        for file in os.listdir(path):
            if file.endswith(".png"):
                mean += np.float(file.split("_")[-2])
                i += 1
        mean = mean / (i + 1e-10)
        print(path,mean)
        f = open(path + "/mean_dist_" + str(np.around(mean,3)) + ".txt", "w+")
        f.close()

