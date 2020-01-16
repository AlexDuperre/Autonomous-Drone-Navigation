import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import time
import socket
import time

import models

import cv2, queue, threading

import pyxhook


key = None
keyPressTime = time.time()
def OnKeyPress(event):
    #print(event.Key)
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
    time.sleep(5)
    cam = VideoCapture('tcp:127.0.0.1:5005') #tcp://192.168.1.1:5555 for real drone camera
    
    # Default input size
    height = 360
    width = 640
    channels = 3
    batch_size = 1
   
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

        plt.ion()
        #fig = plt.figure()
        #fig.show()

        ax1 = plt.subplot(111)
        im1 = ax1.imshow(cam.read())
        ax2 = plt.colorbar(im1)

        running = True
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

            frame = cam.read()
            img = np.array(frame).astype('float32')
            img = np.expand_dims(np.asarray(img), axis=0)

            # Evalute the network for the given image
            pred = sess.run(net.get_output(), feed_dict={input_node: img})
            image = pred[0,:,:,0]

            # Plot result
            # ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')

            #image *=(255.0/image.max())
            im1.set_data(image)
            im1.set_clim(vmin=pred.min(),vmax=pred.max())
            # ax2.set_ticks(np.linspace(pred.min(),pred.max(),num=10))
            # im1.draw_all()
            plt.pause(0.00001)
            # fig.colorbar(ii)
            # plt.draw()
            # plt.pause(0.00001)
            # plt.clf()

            # key processing for training
            global key
            global keyPressTime
            simTime = time.time()
            if (simTime-keyPressTime)>0.5:
                key = None
            #print(key)



            if running:
                if key == 'Escape':
                    # escape key pressed
                    running = False
            else:
                # error reading frame
                print('error reading video feed')


            # Measuring time t2
            t2 = time.time()
            print(t2-t1)
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

        



