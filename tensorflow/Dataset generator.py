import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.pylab import cm
import socket
import time

import models

import cv2, queue, threading

import util.keyloggerFct as keyloggerFct

from util.tools import post_treatment
from util.tools import telemetry_transform
from util.tools import get_goals

import h5py
from util.tools import extract_data
from util.tools import indexer

from util.tools import VideoCapture
"""

This code communicates with the drone interface and collects the image feed of the drone as well as its position.
When launched, it allows to generate individual demonstration trajectories by a human expert in order to create an 
imitation dataset. 

To start recording, first press T while the drone camera window is active. Then press Enter to start recording. 
Once you reached your destination, press Space to stop and save the recording. The next destination will appear in your
depth window. You can start recording again when you are ready by pressing once again Enter. At anytime during the 
recording, you can cancel the current trajectory by pressing Backspace. You can then return to the appropriate starting 
point and press Enter to start over. 
Each trajectory is saved at the specified location in the Datset_generator.py script with the following format: 
path_X_Y_ZZZ.h5. 
Where X stands for the ID of the start point ant Y the ID of the destination. The ZZZ are corresponding to the 
increments for this specific trajectory.

To control de drone, please refer to  the https://github.com/Barahlush/ardrone_autopilot GitHub repo.
The key used to control the drone are W, Q, E, S, A, D, L, U, J


To begin, select the number associated to the world

1 - Luxury home
2-  Luxury home 2nd floor
3 - Bar
4 - Machine room
5 - Mechanical plant
6 - office
7 - Resto bar
8 - Scanned home
9 - Scanned home 2nd floor

Alexandre Duperre - 09/10/2020
"""

world_no = 9
world_strings = ["luxury_home", "luxury_home_2e_floor", "bar", "machine_room", "mechanical_plant", "office", "resto_bar", "scanned_home", "scanned_home_2e_floor"]

# Initialize and start keylogger
keyloggerFct.init()
keywatch = keyloggerFct.Keystroke_Watcher()
keywatch.hm.start()


def predict(model_data_path):
    # Initialize TCP/IP connection
    TCP_IP = '127.0.0.1'
    TCP_PORT = 5007
    BUFFER_SIZE = 1000

    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to position and video feed (in that order)
    s2.connect((TCP_IP, TCP_PORT))
    time.sleep(3)
    cam = VideoCapture('tcp:127.0.0.1:5005')  # tcp://192.168.1.1:5555 for real drone camera

    # Default input size
    height = 360
    width = 640
    channels = 3
    batch_size = 1

    # Define destination for relative coordinates
    destination_list = get_goals("./destinations.ods",world_no)
    destination = eval(destination_list[1])
    last_destination_id = 0
    destination_id = 1



    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        # net.load(model_data_path, sess)

        running = True
        recording = False
        batch = []
        while running:
            # Measuring time t1
            # t1 = time.time()

            # get and decode current frame telemetry
            data = s2.recv(BUFFER_SIZE)
            data = data.decode().split("#")[1] # Safe character to avoid catching other information
            split_data = data.split(',')
            # Makes sure no other data is in the last value
            if len(split_data[3].split('.')) > 2:
                var = split_data[3].split('.')
                split_data[3] = var[0] + '.' + var[1][0:3]

            collision = split_data[4]
            split_data = split_data[0:4]

            # Calculate relative position and arrow angle
            rel_orientation, display_angle, rel_destination, destination_orientation = telemetry_transform(destination, split_data)
            arrowpoint = (int(320 + 40 * np.cos(display_angle)), int(320 - 40 * np.sin(display_angle)))

            # get current frame
            frame = cam.read()
            img = np.array(frame).astype('float32')
            img = np.expand_dims(np.asarray(img), axis=0)

            # Evalute the network for the given image
            pred = sess.run(net.get_output(), feed_dict={input_node: img})
            min = (pred[0, :, :, 0]).min()
            max = (pred[0, :, :, 0]).max()
            image = pred[0, :, :, 0]

            # Add elements to the view (min, max and arrow)
            image = post_treatment(image, rel_destination, arrowpoint, min, max)

            # Append informaion to create dataset along a trajectory
            if recording == True:
                image = cv2.putText(image, 'Recording', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                batch.append([pred[0,:,:,0], img, float("%.3f"%(rel_destination[0])), float("%.3f"%(rel_destination[1])), float("%.3f"%(destination_orientation)), float("%.3f"%(rel_orientation)), np.string_(keyloggerFct.key)])

            # Displays Depth window
            cv2.imshow("Depth", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Prints sent commands to the drone
            print(keyloggerFct.key)

            # if statment for recording flight
            if (keyloggerFct.key == 'space') & (recording == True):

                # Create saving directory if it doesn't exist
                directory = "/tmp/dataset/ardrone_" + world_strings[world_no-1]
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # Create name fore the path between the two destinations and increment it if that path was already done
                title = directory + "/path_" + str(last_destination_id) + "_" + str(destination_id) + "_000.h5"
                title = indexer(title)

                # Save the trajectory
                with h5py.File(title,"w") as hdf:       
                    hdf.create_dataset("depth", data=extract_data(batch, 0), compression="gzip")
                    hdf.create_dataset("video", data=extract_data(batch, 1), compression="gzip")
                    hdf.create_dataset("rel_goalx", data=extract_data(batch, 2), compression="gzip")
                    hdf.create_dataset("rel_goaly", data=extract_data(batch, 3), compression="gzip")
                    hdf.create_dataset("goal_orientation", data=extract_data(batch, 4), compression="gzip")
                    hdf.create_dataset("rel_orientation", data=extract_data(batch, 5), compression="gzip")
                    hdf.create_dataset("GT", data=extract_data(batch, 6), compression="gzip")

                # Refresh trajectory and stop recording
                batch = []
                recording = False
                # save last destination and get new destination
                last_destination_id = destination_id
                destination_id = np.random.randint(len(destination_list))
                # destination_id += 1
                destination = eval(destination_list[destination_id])

                print("RECORDING STOPPED")

            if keyloggerFct.key == 'Return':
                recording = True
                print("RECORDING")

            if keyloggerFct.key == 'BackSpace':
                batch = []
                recording = False
                print("RECORDING DISCARDED")

            if running:
                if keyloggerFct.key == 'Escape':
                    # escape key pressed
                    running = False
            else:
                # error reading frame
                print('error reading video feed')

            # Measuring time t2
            # t2 = time.time()
            # print(t2-t1)
        plt.ioff()
        plt.show()
        plt.close('Figure 1')


def main():
    model_path = "./checkpoint/NYU_FCRN.ckpt"

    # Launch code
    pred = predict(model_path)

    os._exit(0)


if __name__ == '__main__':
    main()

