# Autonomous Drone Guidance
## Introduction
This repo is the result of a Master's thesis on Reactive guidance and trajectory planning for monocular drones using artificial intelligence. 
The objective of this research project is to create an intelligent agent capable of imitating a human guidance policy in a complex and unknown environment based on a depth map image and relative goal inputs. Considering the lower cost in development and computation time, the imitation learning approach was chosen. A sophisticated simulation environment was set up to create an imitation learning datasets. A total of 624 suboptimal demonstration paths from 9 different 3D environments were gathered, which represent 296 466 learning pairs. The demonstrations are qualified as suboptimal since the expert is a human trying its best to solve the guidance problem without any optimal planners.
A classification model was introduced for predicting the appropriate guidance command based on the observations over time. The model learned a meaningful representation of its inputs which can be processed by a long short-term memory network (LSTM) followed by a fully connected network. In this way, the depth image obtained from the RGB original image along with the relative coordinates to the destination are converted into a guidance command at each time step. In order to improve the classification accuracy on the test set, a custom loss function and data augmentation techniques were implemented.
The current model is designed to be used on the Parrot AR Drone 2.0. It is using the well known Tum Simulator as its simulation environment but can control the real drone with the appropriate modifications to the ROS launchfile. However, the model can be reused to control different robots by integrating the guidance model to any desired ROS node.

## Installation
### Dependency
The installation is quite complex. The current setup is running on Ubuntu 16.04 LTS with GPU capabilities (NVIDIA 1080Ti). 
First you'll need to install ROS Kinetic, Gazebo 7.15.06 and Python 3.6.

Make sure you install all the python packages according to the requirements.txt from this repo.

Then, you will need to install the following ROS packages in the same workspace :

 - ardrone_autonomy & tum simulator (tutorial : http://wiki.ros.org/tum_simulator)
 - ardrone_autopilot (Option 1: from https://github.com/Barahlush/ardrone_autopilot, Option 2 : CMake from the ardrone_autopilot-go in this repo)
 
If Option 1 is chosen, once the ardrone_autopilot is installed, swap the file interface.py in the ardrone_autopilot/node directory for the one in the /ardrone_autopilor-go/nodes  directory of this repo. This version of the interface.py file will allow communication between the model running on Python 3.6 and the ROS nodes.
 The new launchfiles will also need to be copied from this repo to the appropriate location in order to use the 3D environments mentioned below. 
 
### Environments 
In order for the simulation to be realistic, 3D models of various environments have been purchased. They can be downloaded via this link: *****. Beware that the licence only allows people from CoSIM to use them for their non-profit purposes. Please refer to the licence documentation before using these files (https://blog.turbosquid.com/turbosquid-3d-model-license/)

The worlds will need to be located to the appropriate location according to the launchfiles. By default, they are expected to be in the following location:

    $HOME./gazebo/models/<example_world>

(Link for the files will be here soon)

## Usage explanation
The main branch of this project will allow you to train the Automated Drone Guidance (AGN) model. In order to do so, you need to download and extract the chosen version of the Drone Guidance Dataset (DGD) and modify the path in the main.py file to be able to access it.

The DGD has 4 versions with the following characteristics:

 - Version 1 : Full raw dataset with "None" classes for when the expert  didn't do anything while recording the trajectories
 - Version 2 : Full size images without "None" classes
 - Version 3 : Lower size images without "None" classes (for RGBVarSeq training)
 - Version 4 : Lower size images without "None" classes and RGB images

### Training
For training, it is recommended to use version 3 except if you have memory issues. In that case, version 4 is more appropriate but slower due to HDF5 parameters chosen.
In any case you should select one environment in the datset for the validation/test phase. We have chosen the environment *Bar* to be our default environment. The chosen environment must be separated form the dataset and placed in another folder next to the training folder. The path to this validation/test environment must be corrected in the main.py file.
(Link will be uploaded soon)

Il you choose to use a pretrained encoder for the depth image, please make sure you have the option turned on in tne main.py and the checkpoint.pt in the /Best_models/Autoencoder/8/ location.
(Link will be uploaded soon)

After setting up the training files, it will be possible to launch the main.py file to begin training. Different hyperparameters can be used by modifying the hyperparm dictionnary in the main.py file.

The training process is recorded via the Comet.ml architecture. You will need to create a Comet .ml account and create a key to log your training. The key must be changed into the first section of the main.py code. It will log the training curves as well as a backup of your main.py and the custom_loss.py file and finally a confuion matrix of your performances in the test phase.

At the end of the training, a model.pt file will appear in the project folder containing the trained weights of your model. They can be loaded into the model for further training or for the guidance simulation.

### Simulation testing and data gathering

The trained model can also directly be used for simulation (or experimental testing if connected to the real drone) or for data gathering. For both cases, you'll need to make sure you are using the Navigator_prev_comm branch. This branch allows the communication between the drone and the model or the keyboard. Depending on if you want to launch a simulation or build your own dataset, you will be interested into the Drone_Navigator.py or the Dataset_generator.py scripts.

The procedure to launch these two modes is the same. The only difference resides in the script you will launch.

Step 1 : Launch the Gazebo world of your choice

    roslaunch cvg_sim_gazebo ardrone_bar.launch

Step 2 : Launch the script corresponding to the chosen mode (via command line or your code editor)

    python3 ./Drone_Navigator.py
    OR
    python3 ./Datset_generator.py

Step 3 : Launch the drone interface

    roslaunch ardrone_autopilot autopilot.launch

Wait until the RGB and the depth Image appears before starting to play with the drone.

#### Simulation and testing

If you used the Drone_Navigator.py script for simulation, the script will allow the drone to reach a destination specified in the script automatically. You only need to press the key T on your keyboard while you have the Drone camera window selected, then press Enter when you are ready to launch the ADG model. At anytime during the flight, you can make the drone land by pressing L, stop the guidance mode by pressing Backspace (then start it with a new point by pressing Enter), or quit by pressing Escape.

#### Data gathering

In the case you wanted to generate data or record a trajectory, you will be ale to start recording by pressing Enter. Once you reached your destination, press Space to stop and save the recording. The next destination will appear in your depth window. You can start recording again when you are ready by pressing once again Enter. At anytime during the recording, you can cancel the current trajectory by pressing Backspace. You can then return to the appropriate starting point and press Enter to start over. 
Each trajectory is saved at the specified location in the Datset_generator.py script with the following format: path_X_Y_ZZZ.h5. Where X stands for the ID of the start point ant Y the ID of the destination. The ZZZ are corresponding to the increments for this specific trajectory.

To control de drone, please refer to the modified table below from the https://github.com/Barahlush/ardrone_autopilot GitHub repo.

|Buttons: |Info|
|-----|------|
|W, A, S, D | tilt forward/left/backward/right
|Q, E| rotate left/right
|T | take off
|L | land
|[ or ] | up or down
|C | change camera

## Link to the pretrained model

(Link will be uploaded soon)
   