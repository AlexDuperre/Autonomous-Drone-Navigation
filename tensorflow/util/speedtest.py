from comet_ml import Experiment
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import lr_scheduler

from models.dataset import DND
from models.model import LSTMModel
from util.focalloss import FocalLoss
from util.custom_loss import weightedLoss

import numpy as np
import matplotlib.pyplot as plt

from util.pytorchtools import EarlyStopping



from sklearn.metrics import confusion_matrix
from util.confusion_matrix import plot_confusion_matrix
# from GPUtil import showUtilization as gpu_usage
import time

hyper_params = {
    "validationRatio" : 0.3,
    "validationTestRatio" : 0.5,
    "batch_size" : 100,
    "learning_rate" : 0.001,
    "specific_lr" : 0.0001,
    "lr_scheduler_step" : 20,
    "num_epochs" : 65,
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


# Initialize the dataset
t1 = time.time()
dataset = DND("C:/aldupd/DND/light dataset None free/", frames_nb=hyper_params["frame_nb"], subsegment_nb=hyper_params["sub_segment_nb"], overlap=hyper_params["segment_overlap"]) #/media/aldupd/UNTITLED 2/dataset
print("temps: ", time.time()-t1)
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
model = LSTMModel(input_dim=hyper_params["input_dim"],
                  hidden_dim=hyper_params["hidden_dim"],
                  layer_dim=hyper_params["layer_dim"],
                  output_dim=hyper_params["output_dim"],
                  Pretrained=False)
model = model.cuda()

# LOSS
#criterion = nn.CrossEntropyLoss(weight=torch.Tensor([3.2046819111, 1, 5.9049048165, 5.4645210478, 15.7675989943, 15.4961006872]).cuda())
# criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.0684208353, 0.0213502735, 0.1260713329, 0.116669019, 0.3366425512, 0.3308459881]).cuda())
# criterion = FocalLoss(gamma=5)
# criterion = nn.CrossEntropyLoss()
criterion = weightedLoss()

# Optimzer
optimizer = torch.optim.Adam([{"params": model.densenet.parameters(), "lr": hyper_params["specific_lr"]},
                              {"params": model.lstm.parameters()},
                              {"params": model.fc.parameters()}],
                             lr=hyper_params["learning_rate"],
                             weight_decay=0.01)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=hyper_params["lr_scheduler_step"], gamma=0.1)

print("model loaded")

trainLoss = []
validLoss = []
validAcc = []
eps = 0.0000000001
best_val_acc = 0
step = 0
total_step = len(train_loader)
for epoch in range(hyper_params["num_epochs"]):
    print("############## Epoch {} ###############".format(epoch+1))
    meanLoss = 0
    sub_segment_nb = 0
    batch_step = 0
    model.train()
    outputs = []
    t0 = time.time()
    for i, (depth, rel_orientation, rel_goalx, rel_goaly, labels) in enumerate(train_loader):
        t1=time.time()
        print(t1-t0)
        t0 = t1

        if i == 10:
            break
