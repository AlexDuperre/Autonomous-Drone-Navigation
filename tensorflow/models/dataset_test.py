import torch
from models.dataset import Dataset


dataset = Dataset("../util/")
print(dataset.__len__())

validationRatio = 0.01
validationTestRatio = 0.5
batch_size = 10
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

for i, (depth, rel_orientation,rel_goalx, rel_goaly) in enumerate(train_loader):
    print(depth)