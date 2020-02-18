import torch
from models.dataset import DND
from models.model import LSTMModel
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

dataset = DND("../util/", frames_nb=50)
print("Dataset length: ", dataset.__len__())

validationRatio = 0.01
validationTestRatio = 0.5
batch_size = 1

# sending to loader
# torch.manual_seed(0)
indices = torch.randperm(len(dataset))
train_indices = indices[:len(indices) - int((validationRatio) * len(dataset))]
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)

valid_train_indices = indices[len(indices) - int(validationRatio * len(dataset)):]
valid_indices = valid_train_indices[:len(valid_train_indices) - int((validationTestRatio) * len(valid_train_indices))]
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

test_indices =  valid_train_indices[len(valid_train_indices) - int(validationTestRatio * len(valid_train_indices)):]
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

# Dataset
train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           sampler = train_sampler,
                                           shuffle=False,
                                           num_workers=0)


model = LSTMModel(input_dim=100, hidden_dim=100, layer_dim=1,output_dim=6)
model = model.cuda()
learning_rate =0.0001
num_epochs = 10

criterion = nn.CrossEntropyLoss()
# criterion = FocalLoss(gamma=4)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


print("model loaded")

trainLoss = []
validLoss = []
validAcc = []
best_val_acc = 0
total_step = len(train_loader)
for epoch in range(num_epochs):
    print("########## Epoch ###########", epoch+1)
    meanLoss = 0
    model.train()
    for i, (depth, rel_orientation, rel_goalx, rel_goaly, labels) in enumerate(train_loader):
        input = [depth.cuda(),
                 torch.cat([rel_orientation.unsqueeze(-1), rel_goalx.unsqueeze(-1), rel_goaly.unsqueeze(-1)], -1).cuda()]

        # Forward pass
        outputs = model(input)

        loss = criterion(outputs.view(-1,6), labels.view(-1).cuda())
        meanLoss += loss.cpu().detach().numpy()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


    # Append mean loss fo graphing and apply lr scheduler
    trainLoss.append(meanLoss / (i + 1))


    # Validation of the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

x = np.linspace(0,num_epochs,num_epochs)
plt.plot(x,trainLoss)
plt.show()