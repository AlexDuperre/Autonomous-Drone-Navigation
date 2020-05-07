from comet_ml import Experiment
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import lr_scheduler

from models.dataset import DND
from models.model import LSTMModel
from util.focalloss import FocalLoss
from util.custom_loss import weightedLoss
from util.custom_loss import pathLoss

import numpy as np
import matplotlib.pyplot as plt

from util.pytorchtools import EarlyStopping

from util.tools import compute_paths
from util.tools import display_paths

from sklearn.metrics import confusion_matrix
from util.confusion_matrix import plot_confusion_matrix
# from GPUtil import showUtilization as gpu_usage


hyper_params = {
    "validationRatio" : 0.3,
    "validationTestRatio" : 0.5,
    "pretrained" : True,
    "batch_size" : 100,
    "learning_rate" : 0.001,
    "specific_lr" : 0.00001,
    "lr_scheduler_step" : 15,
    "num_epochs" : 32,
    "input_dim" : 850,
    "hidden_dim" : 1000,
    "layer_dim" : 1,
    "output_dim" : 5,
    "frame_nb" : 100,
    "sub_segment_nb": 1,
    "segment_overlap": 0,
    "patience" : 20,
    "skip_frames" : 3
}


experiment = Experiment(api_key="rdQN9vxDPj1stp3lh9rIYZfDE",
                        project_name="ADN", workspace="alexduperre")
experiment.log_parameters(hyper_params)


# Initialize the early_stopping object
early_stopping = EarlyStopping(patience=hyper_params["patience"], verbose=True)

# Initialize the dataset
transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1792,), (0.1497,)),
])

dataset = DND("C:/aldupd/DND/Smaller depth None free/", frames_nb=hyper_params["frame_nb"], subsegment_nb=hyper_params["sub_segment_nb"], overlap=hyper_params["segment_overlap"], goal_chg=True) #/media/aldupd/UNTITLED 2/dataset

val_test_set = DND("C:/aldupd/DND/val-test set/", frames_nb=hyper_params["frame_nb"], subsegment_nb=hyper_params["sub_segment_nb"], overlap=hyper_params["segment_overlap"], goal_chg=True) #/media/aldupd/UNTITLED 2/dataset

print("Dataset length: ", dataset.__len__())



# Sending to loader
# train sampler
indices = torch.randperm(len(dataset))
# train_indices = indices[:len(indices) - int((hyper_params["validationRatio"]) * len(dataset))]
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)

indices = torch.randperm(len(val_test_set))
valid_indices = indices[:len(indices) - int(hyper_params["validationRatio"] * len(val_test_set))]
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

test_indices =  indices[len(indices) - int(hyper_params["validationRatio"] * len(val_test_set)):]
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)


train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=hyper_params["batch_size"],
                                           sampler = train_sampler,
                                           shuffle=False,
                                           num_workers=0)
valid_loader = torch.utils.data.DataLoader(dataset=val_test_set,
                                           batch_size=hyper_params["batch_size"],
                                           sampler=valid_sampler,
                                           shuffle=False,
                                           num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=val_test_set,
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
                  Pretrained=hyper_params["pretrained"])
model = model.cuda()

# LOSS
#criterion = nn.CrossEntropyLoss(weight=torch.Tensor([3.2046819111, 1, 5.9049048165, 5.4645210478, 15.7675989943, 15.4961006872]).cuda())
# criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.0684208353, 0.0213502735, 0.1260713329, 0.116669019, 0.3366425512, 0.3308459881]).cuda())
# criterion = FocalLoss(gamma=5)
# criterion = nn.CrossEntropyLoss()
criterion = weightedLoss()
val_loss = pathLoss(frequency=10)

# Optimzer
optimizer = torch.optim.Adam([{"params": model.densenet.parameters(), "lr": hyper_params["specific_lr"]},
                              {"params": model.lstm.parameters()},
                              {"params": model.fc.parameters()},
                              {"params": model.orientation_rep.parameters(), "lr": 0.0001}],
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
    batch_step = 0
    model.train()
    outputs = []
    for i, (depth, rel_orientation, rel_goalx, rel_goaly, labels, lengths, mask) in enumerate(train_loader):
        depth = depth.view(depth.shape[0], hyper_params["frame_nb"], depth.shape[2], depth.shape[3])
        rel_orientation = rel_orientation.view(depth.shape[0], hyper_params["frame_nb"], -1)
        rel_goalx = rel_goalx.view(depth.shape[0], hyper_params["frame_nb"], -1)
        rel_goaly = rel_goaly.view(depth.shape[0], hyper_params["frame_nb"], -1)
        labels = labels.view(depth.shape[0], hyper_params["frame_nb"])

        depth = depth[:, 0:-1:hyper_params["skip_frames"], :, :].requires_grad_() #loses one time step of the segment
        rel_orientation = rel_orientation[:, 0:-1:hyper_params["skip_frames"]].requires_grad_()
        rel_goalx = rel_goalx[:, 0:-1:hyper_params["skip_frames"]].requires_grad_()
        rel_goaly = rel_goaly[:, 0:-1:hyper_params["skip_frames"]].requires_grad_()
        labels = labels[:, 0:-1:hyper_params["skip_frames"]].long()
        mask = mask[:, 0:-1:hyper_params["skip_frames"]]
        lengths = lengths//hyper_params["skip_frames"]

        # mask = torch.zeros([1, 1, 96, 160])
        # mask[0, 0, 32:64, 53:106] = 1
        # mask = mask.cuda()

        inputA = depth.cuda()
        inputB = torch.cat([rel_orientation, torch.sqrt(rel_goalx**2 + rel_goaly**2)], -1).cuda()
        input = [inputA, inputB.float()]
        lengths = lengths.cuda()
        label = labels.cuda()

        # Forward pass
        outputs, (hn, cn) = model(input, lengths)

        # loss = criterion(outputs.view(-1,6), label.view(-1))
        loss = criterion(outputs, label, input, mask)
        # loss = criterion(outputs, label)
        meanLoss += loss.cpu().detach().numpy()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        experiment.log_metric("train_batch_loss", loss.item(), step=step+i+1)


        if (i + 1) % 8 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, hyper_params["num_epochs"], i + 1, total_step, loss.item()))

    step += (i+1)
    # Append mean loss fo graphing and apply lr scheduler
    trainLoss.append(meanLoss / (i + 1))
    experiment.log_metric("train_epoch_loss", meanLoss / (i + 1) , step=epoch)


    # Validation of the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    with torch.no_grad():
        correct = 0
        total = 0
        meanLoss = 0
        meanLoss1 = 0
        for i, (depth, rel_orientation, rel_goalx, rel_goaly, labels, lengths, mask) in enumerate(valid_loader):
            depth = depth.view(depth.shape[0], hyper_params["frame_nb"], depth.shape[2],
                               depth.shape[3])
            rel_orientation = rel_orientation.view(depth.shape[0], hyper_params["frame_nb"], -1)
            rel_goalx = rel_goalx.view(depth.shape[0], hyper_params["frame_nb"], -1)
            rel_goaly = rel_goaly.view(depth.shape[0], hyper_params["frame_nb"], -1)
            labels = labels.view(depth.shape[0], hyper_params["frame_nb"])

            depth = depth[:, 0:-1:hyper_params["skip_frames"], :, :]
            rel_orientation = rel_orientation[:, 0:-1:hyper_params["skip_frames"], :]
            rel_goalx = rel_goalx[:, 0:-1:hyper_params["skip_frames"], :]
            rel_goaly = rel_goaly[:, 0:-1:hyper_params["skip_frames"], :]
            labels = labels[:, 0:-1:hyper_params["skip_frames"]].long()
            mask = mask[:, 0:-1:hyper_params["skip_frames"]]
            lengths = lengths // hyper_params["skip_frames"]

            inputA = depth.cuda()
            inputB = torch.cat([rel_orientation, torch.sqrt(rel_goalx** 2 + rel_goaly** 2)], -1).cuda()
            input = [inputA, inputB.float()]
            label = labels.cuda()

            # Forward pass
            outputs, (hn, cn) = model(input, lengths)

            # loss = criterion(outputs.view(-1, 6), label.view(-1))
            loss1 = criterion(outputs, label, input, mask)
            loss = val_loss(outputs, label, epoch)

            meanLoss += loss.cpu().detach().numpy()
            meanLoss1 += loss1.cpu().detach().numpy()
            _, predicted = torch.max(outputs.data, 2)
            total += len(label.view(-1))
            correct += (predicted.view(-1) == label.view(-1)).sum().item()



        acc = 100 * correct / total
        if acc > best_val_acc:
            best_val_acc = acc

        print('Validation Accuracy : {} %, Loss : {:.4f}'.format(acc, meanLoss / (i+1)))
        validLoss.append(meanLoss / len(valid_loader))
        validAcc.append(acc)
        experiment.log_metric("valid_epoch_loss", meanLoss / (i+1), step=epoch)
        experiment.log_metric("valid_epoch_loss1", meanLoss1 / (i+1), step=epoch)
        experiment.log_metric("valid_epoch_accuracy", acc, step=epoch)

    # Adjust learning rate
    # if epoch < 20:
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
    for i, (depth, rel_orientation, rel_goalx, rel_goaly, labels, lengths, mask) in enumerate(test_loader):
        depth = depth.view(depth.shape[0], hyper_params["frame_nb"], depth.shape[2],depth.shape[3])
        rel_orientation = rel_orientation.view(depth.shape[0], hyper_params["frame_nb"], -1)
        rel_goalx = rel_goalx.view(depth.shape[0], hyper_params["frame_nb"], -1)
        rel_goaly = rel_goaly.view(depth.shape[0], hyper_params["frame_nb"], -1)
        labels = labels.view(depth.shape[0], hyper_params["frame_nb"])

        depth = depth[:, 0:-1:hyper_params["skip_frames"], :, :]
        rel_orientation = rel_orientation[:, 0:-1:hyper_params["skip_frames"], :]
        rel_goalx = rel_goalx[:, 0:-1:hyper_params["skip_frames"], :]
        rel_goaly = rel_goaly[:, 0:-1:hyper_params["skip_frames"], :]
        labels = labels[:, 0:-1:hyper_params["skip_frames"]].long()
        mask = mask[:, 0:-1:hyper_params["skip_frames"]]
        lengths = lengths // hyper_params["skip_frames"]

        inputA = depth.cuda()
        inputB = torch.cat([rel_orientation, torch.sqrt(rel_goalx**2 + rel_goaly**2)], -1).cuda()
        input = [inputA, inputB.float()]

        label = labels.cuda()

        # Forward pass
        outputs, (hn, cn) = model(input, lengths)


        # loss = criterion(outputs.view(-1,6), label.view(-1))
        loss = criterion(outputs, label, input, mask)
        meanLoss += loss.cpu().detach().numpy()

        _, predicted = torch.max(outputs.data, 2)

        predictions = np.append(predictions,predicted.view(-1).cpu().detach().numpy())
        ground_truth = np.append(ground_truth,label.view(-1).cpu().detach().numpy())

        total += len(label.view(-1))
        correct += (predicted.view(-1) == label.view(-1)).sum().item()


    test_acc = 100 * correct / total
    print('Test Accuracy : {} %, Loss : {:.4f}'.format(test_acc, meanLoss / (i+1)))

# Logging reults
experiment.log_metric("test_loss", meanLoss / (i+1), step=epoch)
experiment.log_metric("test_accuracy", test_acc, step=epoch)

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
plt.figure()
cm = confusion_matrix(ground_truth,predictions)
plot_confusion_matrix(cm.astype(np.int64), classes=[ "w", "q", "e", "w+q", "w+e"], path=".")

experiment.log_image("./confusion_matrix.png")


dict = {
    "test_acc" : test_acc
}

experiment.send_notification("finished", "ok tamere", dict)