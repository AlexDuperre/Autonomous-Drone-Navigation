import torch
import torch.nn as nn
from util.tools import display_paths
class weightedLoss(nn.Module):
    def __init__(self):
        super(weightedLoss, self).__init__()
        self.crossentropy = nn.CrossEntropyLoss(weight=torch.Tensor([0.8213502735, 3, 2, 4.3366425512, 4.3308459881]).cuda(),reduction="none") #[0.0684208353, 0.0213502735, 0.1260713329, 0.116669019, 0.3366425512, 0.3308459881]

    def sample_weighter(self, depth):
        batch, frame_nb, _, _ = depth.shape
        medians, _ = depth.view(batch, frame_nb, -1).median(dim=2)
        cond = medians <= 0.16
        weights = torch.zeros(medians.shape).cuda()
        weights[cond] = torch.ones(medians.shape)[cond].cuda() - torch.div(medians[cond],0.2)
        weights += torch.ones(medians.shape).cuda()
        weights = weights**4
        weights = weights / weights.sum()
        return weights.cuda()


    def forward(self, outputs, targets, input):
        weights = self.sample_weighter(input[0]).view(-1)

        Distance_loss  = 2.0*torch.sqrt(input[1][:,:,1]**2 + input[1][:,:,2]**2).view(-1)

        Orientation_loss = 2.0*(input[1][:,:,0]**2).view(-1)

        losses = self.crossentropy(outputs.view(-1, outputs.shape[-1]), targets.view(-1)) + Distance_loss + Orientation_loss
        loss = (losses * weights).sum()

        return loss

class pathLoss(nn.Module):
    def __init__(self, frequency=10):
        super(pathLoss, self).__init__()
        self.fequency = frequency
        self.first_batch = True
        self.current_epoch = 0
    def forward(self, outputs, labels, save=False):
        _, predicted = torch.max(outputs.data, 2)
        predPts = compute_paths_means(predicted)
        truePts = compute_paths_means(labels)

        # Save 20 images of path if current epoch is a multiple of self.frequency AND first batch
        if (save % self.fequency == 0) & (self.first_batch):
            for i in range(0,100,5):
                display_paths(
                    [[predPts[0][i].tolist(), predPts[1][i].tolist()], [truePts[0][i].tolist(), truePts[1][i].tolist()]],
                    save,
                    i)
            self.first_batch = False
            self.current_epoch = save

        # reset criteria for first batch when current epoch is done
        if save != self.current_epoch:
            self.first_batch = True



        loss = torch.sqrt((predPts[0]-truePts[0])**2 + (predPts[0]-truePts[0])**2).mean()
        return loss

"""
Compute path data for the Display_path function 
"""
def compute_paths_means(predictions, dtheta=0.15):
    batch_size = predictions.shape[0]
    pred_points_x = [torch.zeros(batch_size).cuda()]
    pred_points_y = [torch.zeros(batch_size).cuda()]
    dx = torch.zeros(batch_size).cuda()
    dy = torch.zeros(batch_size).cuda()
    theta = torch.zeros(batch_size).cuda()
    for i in range(len(predictions[0])):
         # if predictions[:,i] == 0:
        dx[predictions[:,i] == 0] = torch.cos(theta)[predictions[:,i] == 0]
        dy[predictions[:,i] == 0]  = torch.sin(theta)[predictions[:,i] == 0]
         # if predictions[:,i] == 1:
        theta[predictions[:,i] == 1] = theta[predictions[:,i] == 1] + dtheta
        # if predictions[:,i] == 2:
        theta[predictions[:,i] == 2] = theta[predictions[:,i] == 2] - dtheta
        # if predictions[:,i] == 3:
        theta[predictions[:,i] == 3] = theta[predictions[:,i] == 3] + dtheta
        dx[predictions[:,i] == 3] = torch.cos(theta)[predictions[:,i] == 3]
        dy[predictions[:,i] == 3] = torch.sin(theta)[predictions[:,i] == 3]
        # if predictions[:,i] == 4:
        theta[predictions[:,i] == 4] = theta[predictions[:,i] == 4] - dtheta
        dx[predictions[:,i] == 4] = torch.cos(theta)[predictions[:,i] == 4]
        dy[predictions[:,i] == 4] = torch.sin(theta)[predictions[:,i] == 4]

        pred_points_x.append(pred_points_x[i] + dx)
        pred_points_y.append(pred_points_y[i] + dy)
        dx = torch.zeros(batch_size).cuda()
        dy = torch.zeros(batch_size).cuda()
    return  [torch.stack(pred_points_x, dim=1),torch.stack(pred_points_y, dim=1)]