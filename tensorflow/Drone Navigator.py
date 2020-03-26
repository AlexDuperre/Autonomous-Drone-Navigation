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

import torch
from Best_models.new_best.models.model import LSTMModel

import pyautogui
import subprocess

"""
Select the number associated to the world

1 - Luxury home
2-  Luxury home 2nd floor
3 - Bar
4 - Machine room
5 - Mechanical plant
6 - office
7 - Resto bar
8 - Scanned home
9 - Scanned home 2nd floor
"""

world_no = 9
world_strings = ["luxury_home", "luxury_home_2e_floor", "bar", "machine_room", "mechanical_plant", "office", "resto_bar", "scanned_home", "scanned_home_2e_floor"]

# Initialize and start keylogger
keyloggerFct.init()
keywatch = keyloggerFct.Keystroke_Watcher()
keywatch.hm.start()

# LSTM Hyperparameters
hyper_params = {
    "validationRatio" : 0.3,
    "validationTestRatio" : 0.5,
    "batch_size" : 100,
    "learning_rate" : 0.01,
    "specific_lr" : 0.001,
    "lr_scheduler_step" : 12,
    "num_epochs" : 45,
    "input_dim" : 450,
    "hidden_dim" : 1000,
    "layer_dim" : 1,
    "output_dim" : 5,
    "frame_nb" : 100,
    "sub_segment_nb": 1,
    "segment_overlap": 0,
    "patience" : 10,
    "skip_frames" : 3
}

keys_dict = {
    0 : "w",
    1 : "q",
    2 : "e",
    3 : ["w", "q"],
    4 : ["w", "e"]
}

def predict(model_data_path):
    TCP_IP = '127.0.0.1'
    TCP_PORT = 5007
    BUFFER_SIZE = 1000

    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect in that order
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

    # Add Pytorch Navigator network

    model = LSTMModel(input_dim=hyper_params["input_dim"],
                  hidden_dim=hyper_params["hidden_dim"],
                  layer_dim=hyper_params["layer_dim"],
                  output_dim=hyper_params["output_dim"],
                  Pretrained=False)

    state_dict = torch.load("./Best_models/new_best/checkpoint.pt")
    model.load_state_dict(state_dict)

    model = model
    model.eval()

    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        # net.load(model_data_path, sess)

        # Initialize hidden state with zeros
        hn = torch.zeros(hyper_params["layer_dim"], 1, hyper_params["hidden_dim"]).requires_grad_()
        # Initialize cell state
        cn = torch.zeros(hyper_params["layer_dim"], 1, hyper_params["hidden_dim"]).requires_grad_()

        running = True
        auto_navigating = False
        calls = 0
        start = True
        dt = 0
        t0 = 1e-18
        while running:
            # Measuring time t1
            # t1 = time.time()

            # get current frame telemetry
            data = s2.recv(BUFFER_SIZE)
            split_data = data.decode().split(',')
            if len(split_data[3].split('.')) > 2:
                var = split_data[3].split('.')
                split_data[3] = var[0] + '.' + var[1][0:3]

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

            if auto_navigating == True:
                image = cv2.putText(image, 'Auto Navigating', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                # batch.append([pred[0,:,:,0], img, float("%.3f"%(rel_destination[0])), float("%.3f"%(rel_destination[1])), float("%.3f"%(destination_orientation)), float("%.3f"%(rel_orientation)), np.string_(keyloggerFct.key)])

                calls += 1

                if calls % 1 == 0:
                    depth = cv2.resize(pred[0,:,:,0], dsize=(160, 92), interpolation=cv2.INTER_CUBIC)
                    lstm_input = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
                    out, (hn, cn) = model([lstm_input, 0], hn, cn)
                    _, predicted = torch.max(out.data, 2)
                    predicted = predicted.numpy()[0][0]

                    command = keys_dict[predicted]

                    # makes sure we are in the drone commande window
                    subprocess.call(['./activate_window.sh'])


                    if predicted == 0:
                        print("w")
                        pyautogui.keyDown(command)

                        # if start:
                        #     t0 = time.time()
                        #     start = False

                    elif any(predicted == [1, 2]):
                        print(command)
                        pyautogui.keyUp("w")
                        pyautogui.keyDown(command, pause=2)
                        pyautogui.keyUp(command)

                        # dt = time.time() - t0

                    elif predicted == 3:
                        print("w+q")
                        # if not start:
                        #     pyautogui.keyDown("w")
                        pyautogui.keyDown("q", pause=0.5)
                        pyautogui.keyUp("q")

                        # if not start:
                        #     pyautogui.keyUp("w")

                        # dt = time.time() - t0

                    elif predicted == 4:
                        print("w+e")
                        # if not start:
                        #     pyautogui.keyDown("w")
                        pyautogui.keyDown("e", pause=0.5)
                        pyautogui.keyUp("e")

                        # if not start:
                        #     pyautogui.keyUp("w")
                        #
                        # dt = time.time() - t0

                    # Refresh the hidden state (to be deactivated for long sequences)
                    if calls % 30 == 0:
                        # Initialize hidden state with zeros
                        hn = torch.zeros(hyper_params["layer_dim"], 1, hyper_params["hidden_dim"]).requires_grad_()
                        # Initialize cell state
                        cn = torch.zeros(hyper_params["layer_dim"], 1, hyper_params["hidden_dim"]).requires_grad_()

                        print("Reresh hidden state")

                    # if dt > 5.:
                    #     pyautogui.keyUp("w")
                    #     start = True
                    #     print("Reached forward time limit")
                    #     t0 = 1e-18

                    # print(dt)


            cv2.imshow("Depth", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):

                break

            # print(keyloggerFct.key)

            # if statment for auto_navigating flight
            if (keyloggerFct.key == 'space') & (auto_navigating == True):


                batch = []
                auto_navigating = False
                # save last destination and get new destination
                last_destination_id = destination_id
                destination_id = np.random.randint(len(destination_list))
                # destination_id += 1
                destination = eval(destination_list[destination_id])

                # Initialize hidden state with zeros
                hn = torch.zeros(hyper_params["layer_dim"], 1, hyper_params["hidden_dim"]).requires_grad_()
                # Initialize cell state
                cn = torch.zeros(hyper_params["layer_dim"], 1, hyper_params["hidden_dim"]).requires_grad_()

                print("RECORDING STOPPED")

            if keyloggerFct.key == 'Return':
                pyautogui.keyUp("w")
                pyautogui.keyUp("q")
                pyautogui.keyUp("e")
                auto_navigating = True
                print("RECORDING")

            if keyloggerFct.key == 'BackSpace':
                pyautogui.keyUp("w")
                pyautogui.keyUp("q")
                pyautogui.keyUp("e")
                auto_navigating = False
                print("RECORDING DISCARDED")

            if running:
                if keyloggerFct.key == 'Escape':
                    # escape key pressed
                    pyautogui.keyUp("w")
                    pyautogui.keyUp("q")
                    pyautogui.keyUp("e")
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

    # Predict the image
    pred = predict(model_path)

    os._exit(0)


if __name__ == '__main__':
    main()

