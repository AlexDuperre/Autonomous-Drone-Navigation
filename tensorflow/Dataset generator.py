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

# Initialize and start keylogger
keyloggerFct.init()
keywatch = keyloggerFct.Keystroke_Watcher()
keywatch.hm.start()


# bufferless VideoCapture functions
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
    goal = (-12, -7)

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
        batch_num = 1
        while running:
            # Measuring time t1
            t1 = time.time()

            # get current frame telemetry
            data = s2.recv(BUFFER_SIZE)
            split_data = data.decode().split(',')
            if len(split_data[3].split('.')) > 2:
                var = split_data[3].split('.')
                split_data[3] = var[0] + '.' + var[1][0:3]

            # Calculate relative position and arrow angle
            rel_orientation, display_angle, rel_goal = telemetry_transform(goal, split_data)
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
            image = post_treatment(image, rel_goal, arrowpoint, min, max)

            if recording == True:
                image = cv2.putText(image, 'Recording', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                batch.append([pred[0,:,:,0], float("%.3f"%(rel_goal[0])), float("%.3f"%(rel_goal[1])), float("%.3f"%(goal_orientation)), float("%.3f"%(rel_orientation))])

            cv2.imshow("Depth", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(keyloggerFct.key)

            if running:
                if keyloggerFct.key == 'Escape':
                    # escape key pressed
                    running = False
            else:
                # error reading frame
                print('error reading video feed')

            # Measuring time t2
            t2 = time.time()
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

import pyxhook


key = None
keyPressTime = time.time()
def OnKeyPress(event):
    global key
    key = event.Key
    global keyPressTime
    keyPressTime = time.time()

# Initialize and start keylogger
keylogger = pyxhook.HookManager()
keylogger.KeyDown = OnKeyPress
keylogger.HookKeyboard()
keylogger.start()

# bufferless VideoCapture functions
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
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

def predict(model_data_path):

    TCP_IP = '127.0.0.1'
    TCP_PORT = 5007
    BUFFER_SIZE = 1000

    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect in that order
    s2.connect((TCP_IP, TCP_PORT))
    time.sleep(3)
    cam = VideoCapture('tcp:127.0.0.1:5005') #tcp://192.168.1.1:5555 for real drone camera
    
    # Default input size
    height = 360
    width = 640
    channels = 3
    batch_size = 1

    # Define destination for relative coordinates
    goal = (-12,-7)
    goal_list = [(-12,-7),(0,0),(-16,8)]

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
        #net.load(model_data_path, sess)

        running = True
        recording = False
        batch = []
        batch_num = 1
        while running:
            # Measuring time t1
            t1 = time.time()

            # get current frame of video
            data = s2.recv(BUFFER_SIZE)
            split_data = data.decode().split(',')
            if len(split_data[3].split('.')) > 2:
                var = split_data[3].split('.')
                split_data[3] = var[0] + '.' + var[1][0:3]
            #print(split_data[0:4])

            # Calculate relative position
            try:
                rel_goal = (goal[0]-float(split_data[0]), goal[1]-float(split_data[1]))
            except:
                pass
            goal_orientation = np.arctan2(rel_goal[1], rel_goal[0])
            current_orientation = float(split_data[3])
            if goal_orientation < 0:
                goal_orientation = goal_orientation + 2*np.pi
            if current_orientation < 0:
                current_orientation = current_orientation + 2*np.pi
            rel_orientation = -1*(current_orientation - goal_orientation)
            print("%.3f"%(rel_goal[0]), "%.3f"%(rel_goal[1]), "%.3f"%(goal_orientation), "%.3f"%(rel_orientation))
            display_angle = rel_orientation+(np.pi/2)


            arrowpoint = (int(320 + 40*np.cos(display_angle)), int(320 - 40*np.sin(display_angle)))

            frame = cam.read()
            img = np.array(frame).astype('float32')
            img = np.expand_dims(np.asarray(img), axis=0)

            # Evalute the network for the given image
            pred = sess.run(net.get_output(), feed_dict={input_node: img})
            image = pred[0,:,:,0]

            # Plot result
            image = ((image - image.min()) * (1 / (6 - 0) * 255)).astype('uint8')
            image = np.uint8(cm.jet(image)*255)
            image = cv2.cvtColor(image,cv2.COLOR_RGBA2BGR)
            image = cv2.resize(image, (int(640), int(360)))
            image = cv2.arrowedLine(image, (320,320), arrowpoint, (0,0,255), 2, cv2.LINE_AA)
            text = "%.3f"%((pred[0,:,:,0]).min()) + "m." + "                        " + "%.3f"%((pred[0,:,:,0]).max()) + "m."
            image = cv2.putText(image, "%.3f"%(np.sqrt(rel_goal[0]**2 + rel_goal[1]**2))+'m.', (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            image = cv2.putText(image, text, (0,320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            if recording == True:
                image = cv2.putText(image, 'Recording', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                batch.append([pred[0,:,:,0], float("%.3f"%(rel_goal[0])), float("%.3f"%(rel_goal[1])), float("%.3f"%(goal_orientation)), float("%.3f"%(rel_orientation))])
            cv2.imshow("Depth", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


            # key processing for training
            global key
            global keyPressTime
            simTime = time.time()
            if (simTime-keyPressTime)>0.5:
                key = None
            #print(key)

            # if statment for recording flight
            if (key == 'space') & (recording==True):
                recording = False
                goal = goal_list[np.random.randint(len(goal_list))]
                key = None
                title = "Batch_"+str(batch_num)+".npy"
                np.save(title,batch)
                batch = []
                print("RECORDING STOPPED")
            if key == 'space':
                recording = True
                key = None
                print("RECORDING")






            if running:
                if key == 'Escape':
                    # escape key pressed
                    running = False
            else:
                # error reading frame
                print('error reading video feed')


            # Measuring time t2
            t2 = time.time()
            #print(t2-t1)
        plt.ioff()
        plt.show()
        plt.close('Figure 1')
        # return pred
        
                
def main():
    model_path = "./checkpoint/NYU_FCRN.ckpt"
    # Predict the image
    pred = predict(model_path)

    os._exit(0)

if __name__ == '__main__':
    main()

        



