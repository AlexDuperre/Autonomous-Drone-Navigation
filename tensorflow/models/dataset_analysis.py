import torch
from models.dataset import DND
from models.model import LSTMModel
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

This program needs to run with the original non-augmented dataset. Add the number of dropped frames to the final number 
of frames to get the total in the original dataset. Or modify the dataset to get all frames.

"""
hyper_params = {
    "validationRatio" : 0.3,
    "validationTestRatio" : 0.5,
    "batch_size" : 100,
    "learning_rate" : 0.01,
    "specific_lr" : 0.00001,
    "lr_scheduler_step" : 7,
    "num_epochs" : 1,
    "input_dim" : 150,
    "hidden_dim" : 500,
    "layer_dim" : 1,
    "output_dim" : 5,
    "frame_nb" : 80,
    "sub_segment_nb": 1,
    "segment_overlap": 0,
    "patience" : 10,
    "skip_frames" : 1
}

dataset = DND("/windows/aldupd/val-test set", frames_nb=hyper_params["frame_nb"], subsegment_nb=hyper_params["sub_segment_nb"], overlap=hyper_params["segment_overlap"])
print("Dataset length: ", dataset.__len__())


batch_size = 1

# sending to loader
# torch.manual_seed(0)
train_indices = torch.randperm(len(dataset))
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)



# Dataset
train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           sampler = train_sampler,
                                           shuffle=False,
                                           num_workers=0)

print(len(train_loader))

frame_count = 0
mean = 0
std = 0
classes = np.asarray([0, 0, 0, 0, 0])
for epoch in range(hyper_params["num_epochs"]):
    print("########## Epoch ###########", epoch+1)

    for i, (depth, rel_orientation, rel_goalx, rel_goaly, labels) in enumerate(train_loader):
        print("Batch # ", (i+1))
        # Nb of items in dataset
        frame_count += depth.shape[1]

        # stats of depth
        mean += depth.mean()
        std += depth.std()

        # Count classes
        onehot = np.zeros((depth.shape[1],5))
        onehot[np.arange(depth.shape[1]), np.int8(labels.numpy())] = 1
        classes = classes + onehot.sum(axis=0)

    mean = mean/(i+1)
    std = std/(i+1)

print("Number of frames = ",frame_count)
print("Depth mean = ", mean)
print("Depth std = ", std)
print(classes)

