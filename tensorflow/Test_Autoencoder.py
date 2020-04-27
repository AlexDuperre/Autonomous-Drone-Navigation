from comet_ml import Experiment
import torch
from models.dataset import DND
from models.model import AutoEncoder
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from util.pytorchtools import EarlyStopping
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from util.focalloss import FocalLoss
from sklearn.metrics import confusion_matrix
from util.confusion_matrix import plot_confusion_matrix
from GPUtil import showUtilization as gpu_usage


hyper_params = {
    "validationRatio" : 0.3,
    "validationTestRatio" : 0.5,
    "batch_size" : 1,
    "learning_rate" : 0.0001,
    "lr_scheduler_step" : 9,
    "num_epochs" : 20,
    "frame_nb" : 100,
    "sub_segment_nb": 1,
    "segment_overlap": 0,
    "patience" : 14,
}





# Initialize the dataset
# dataset = DND("/media/aldupd/UNTITLED 2/Smaller depth None free", frames_nb=hyper_params["frame_nb"], subsegment_nb=hyper_params["sub_segment_nb"], overlap=hyper_params["segment_overlap"]) #/media/aldupd/UNTITLED 2/dataset
dataset = DND("C:/aldupd/DND/val-test set", frames_nb=hyper_params["frame_nb"], subsegment_nb=hyper_params["sub_segment_nb"], overlap=hyper_params["segment_overlap"])
print("Dataset length: ", dataset.__len__())



# Sending to loader
# torch.manual_seed(0)
indices = torch.randperm(len(dataset))
train_indices = indices[:len(indices) - int((hyper_params["validationRatio"]) * len(dataset))]
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)

valid_train_indices = indices[len(indices) - int(hyper_params["validationRatio"] * len(dataset)):]
valid_indices = valid_train_indices[:len(valid_train_indices) - int((hyper_params["validationTestRatio"]) * len(valid_train_indices))]
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

test_indices =  valid_train_indices[len(valid_train_indices) - int(hyper_params["validationTestRatio"] * len(valid_train_indices)):]
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)


train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=hyper_params["batch_size"],
                                           sampler = train_sampler,
                                           shuffle=False,
                                           num_workers=0)
valid_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=hyper_params["batch_size"],
                                           sampler=valid_sampler,
                                           shuffle=False,
                                           num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=hyper_params["batch_size"],
                                          sampler=test_sampler,
                                          shuffle=False,
                                          num_workers=0)
print(len(train_loader))
print(len(valid_loader))
print(len(test_loader))

# MODEL
model = AutoEncoder()

model_state_dict = torch.load("./Best_models/Autoencoder/5/checkpoint.pt")

model.load_state_dict((model_state_dict))
model = model.cuda()


print("model loaded")



for i, (depth, rel_orientation, rel_goalx, rel_goaly, labels) in enumerate(train_loader):
    batch, frame_nb, _, _ = depth.shape
    depth = depth.view(batch * frame_nb, depth.shape[2], depth.shape[3]).unsqueeze(1).cuda()


    # Forward pass
    outputs = model(depth)

    outputs = outputs.detach().cpu().numpy()
    depth = depth.detach().cpu().numpy()

    plt.subplot2grid((2,1),(0,0))
    plt.imshow(depth[1][0])

    plt.subplot2grid((2, 1), (1, 0))
    plt.imshow(outputs[1][0])

    plt.show()








