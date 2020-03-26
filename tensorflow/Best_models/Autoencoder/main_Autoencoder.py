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
    "learning_rate" : 0.001,
    "lr_scheduler_step" : 4,
    "num_epochs" : 15,
    "frame_nb" : 100,
    "sub_segment_nb": 1,
    "segment_overlap": 0,
    "patience" : 14,
}


experiment = Experiment(api_key="rdQN9vxDPj1stp3lh9rIYZfDE",
                        project_name="ADN-AutoEncoder", workspace="alexduperre")
experiment.log_parameters(hyper_params)


# Initialize the early_stopping object
early_stopping = EarlyStopping(patience=hyper_params["patience"], verbose=True)

# Initialize the dataset
dataset = DND("/media/aldupd/UNTITLED 2/Smaller depth None free", frames_nb=hyper_params["frame_nb"], subsegment_nb=hyper_params["sub_segment_nb"], overlap=hyper_params["segment_overlap"]) #/media/aldupd/UNTITLED 2/dataset
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
model = model.cuda()

# LOSS
#criterion = nn.CrossEntropyLoss(weight=torch.Tensor([3.2046819111, 1, 5.9049048165, 5.4645210478, 15.7675989943, 15.4961006872]).cuda())
# criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.8, 1.1260713329, 1.116669019, 4.3366425512, 4.3308459881]).cuda())
# criterion = FocalLoss(gamma=5)
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["learning_rate"], weight_decay=0.01)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=hyper_params["lr_scheduler_step"], gamma=0.1)

print("model loaded")

trainLoss = []
validLoss = []
validAcc = []
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
    for i, (depth, rel_orientation, rel_goalx, rel_goaly, labels) in enumerate(train_loader):
        batch, frame_nb, _, _ = depth.shape
        depth = depth.view(batch * frame_nb, depth.shape[2], depth.shape[3]).unsqueeze(1).cuda()
        labels = labels.view(batch * frame_nb,).cuda()


        # Forward pass
        outputs = model(depth)


        loss = criterion(outputs, depth)
        meanLoss += loss.cpu().detach().numpy()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        experiment.log_metric("train_batch_loss", loss.item(), step=step+i+1)


        if (i + 1) % 1000 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, hyper_params["num_epochs"], i + 1, total_step, loss.item()))

    step += i + 1
    # Append mean loss fo graphing and apply lr scheduler
    trainLoss.append(meanLoss / (i + 1))
    experiment.log_metric("train_epoch_loss", meanLoss / (i+1) , step=epoch)


    # Validation of the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    with torch.no_grad():
        correct = 0
        total = 0
        meanLoss = 0
        for i, (depth, rel_orientation, rel_goalx, rel_goaly, labels) in enumerate(valid_loader):
            batch, frame_nb, _, _ = depth.shape
            depth = depth.view(batch * frame_nb, depth.shape[2], depth.shape[3]).unsqueeze(1).cuda()
            labels = labels.view(batch * frame_nb, ).cuda()


            # Forward pass
            outputs = model(depth)

            loss = criterion(outputs, depth)
            meanLoss += loss.cpu().detach().numpy()
            # _, predicted = torch.max(outputs.data, 1)
            # total += len(labels.view(-1))
            # correct += (predicted.view(-1) == labels.view(-1)).sum().item()



        # acc = 100 * correct / total
        # if acc > best_val_acc:
        #     best_val_acc = acc

        print('Validation Loss : {:.4f}'.format(meanLoss / (i+1)))
        validLoss.append(meanLoss / len(valid_loader))
        # validAcc.append(acc)
        experiment.log_metric("valid_epoch_loss", meanLoss / (i+1), step=epoch)
        # experiment.log_metric("valid_epoch_accuracy", acc, step=epoch)

    # Adjust learning rate
    exp_lr_scheduler.step()

    # Check if we should stop early
    early_stopping(meanLoss / (i+1), model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

print("Running on test set")
with torch.no_grad():
    correct = 0
    total = 0
    meanLoss = 0
    predictions = np.empty((0,1))
    ground_truth = np.empty((0,1))
    for i, (depth, rel_orientation, rel_goalx, rel_goaly, labels) in enumerate(test_loader):
        batch, frame_nb, _, _ = depth.shape
        depth = depth.view(batch * frame_nb, depth.shape[2], depth.shape[3]).unsqueeze(1).cuda()
        labels = labels.view(batch * frame_nb, ).cuda()

        # Forward pass
        outputs = model(depth)


        loss = criterion(outputs, depth)
        meanLoss += loss.cpu().detach().numpy()

        # _, predicted = torch.max(outputs.data, 1)

        # predictions = np.append(predictions,predicted.view(-1).cpu().detach().numpy())
        # ground_truth = np.append(ground_truth,labels.view(-1).cpu().detach().numpy())

        total += len(labels.view(-1))
        # correct += (predicted.view(-1) == labels.view(-1)).sum().item()

        sub_segment_nb += 1

    # test_acc = 100 * correct / total
    print('Test Loss : {:.4f}'.format( meanLoss / (i+1)))

# Logging reults
experiment.log_metric("test_loss", meanLoss / (i+1), step=epoch)
# experiment.log_metric("test_accuracy", acc, step=epoch)

# # plotting graphs (not needed if using comet ml)
# plt.figure()
# x = np.linspace(0,hyper_params["num_epochs"],hyper_params["num_epochs"])
# plt.subplot(1,2,1)
# plt.plot(x,trainLoss)
# plt.plot(x,validLoss)
#
# plt.subplot(1,2,2)
# plt.plot(x,validAcc)
# plt.savefig(path+'/learning_curve.png')
# plt.show()

# Plotting confusion matrix
# plt.figure()
# cm = confusion_matrix(ground_truth,predictions)
# plot_confusion_matrix(cm.astype(np.int64), classes=["None", "w", "q", "e", "w+q", "w+e"], path=".")
#
# experiment.log_image("./confusion_matrix.png")


# dict = {
#     "test_acc" : test_acc
# }

experiment.send_notification("finished", "ok tamere", dict)