import torch
import torch.nn as nn
from util.tools import display_paths


class weightedLoss(nn.Module):
    def __init__(self):
        super(weightedLoss, self).__init__()
        self.crossentropy = nn.CrossEntropyLoss(weight=torch.Tensor([0.8213502735, 3, 2.116669019, 4.3366425512, 4.3308459881]).cuda(),reduction="none") #[0.0684208353, 0.0213502735, 0.1260713329, 0.116669019, 0.3366425512, 0.3308459881]

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

        Orientation_loss = 1.0*(input[1][:,:,0]**2).view(-1)

        losses = self.crossentropy(outputs.view(-1, outputs.shape[-1]), targets.view(-1)) + Distance_loss + Orientation_loss
        loss = (losses * weights).sum()

        return loss

class pathLoss(nn.Module):
    def __init__(self, frequency=10):
        super(pathLoss, self).__init__()
        self.fequency = frequency
        self.first_batch = True
        self.current_epoch = 0
    def forward(self, outputs, labels, save=123):
        _, predicted = torch.max(outputs.data, 2)
        mask = torch.zeros_like(outputs).scatter_(2,predicted.unsqueeze(2),torch.ones_like(outputs))
        soft_predictions = torch.softmax(outputs,2)*mask
        predPts = compute_paths_means(soft_predictions)
        truePts = compute_paths_means(torch.zeros_like(outputs).scatter_(2,labels.unsqueeze(2),torch.ones_like(outputs)))

        # Save 20 images of path if current epoch is a multiple of self.frequency AND first batch
        if (save % self.fequency == 0) & (self.first_batch):
            for i in range(0,predicted.shape[0],5):
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
    pred_points_x = [torch.zeros(batch_size).requires_grad_().cuda()]
    pred_points_y = [torch.zeros(batch_size).requires_grad_().cuda()]
    theta = torch.zeros(batch_size).requires_grad_().cuda()
    for i in range(len(predictions[0])):
        pred_x_t = torch.zeros(batch_size).cuda()
        pred_y_t = torch.zeros(batch_size).cuda()
        for j in range(batch_size):
            matrix = torch.Tensor([[torch.cos(theta[j]),torch.sin(theta[j]), 0],
                                   [0, 0, dtheta],
                                   [0, 0, -dtheta],
                                   [torch.cos(theta[j]), torch.sin(theta[j]), dtheta],
                                   [torch.cos(theta[j]), torch.sin(theta[j]), -dtheta]]).cuda()
            sum = ((predictions[j,i]/(predictions[j,i] + 0.000001)).repeat(3).reshape(3,5)*matrix.transpose(1,0)).sum(1)
            pred_x_t[j] = pred_points_x[i][j] + sum[0]
            pred_y_t[j] = pred_points_y[i][j] + sum[1]
            theta[j] = theta[j] + sum[2]

        pred_points_x.append(pred_x_t)
        pred_points_y.append(pred_y_t)

    return  [torch.stack(pred_points_x, dim=1),torch.stack(pred_points_y, dim=1)]


    # batch_size = predictions.shape[0]
    # pred_points_x = [torch.zeros(batch_size).requires_grad_().cuda()]
    # pred_points_y = [torch.zeros(batch_size).requires_grad_().cuda()]
    # dx = torch.zeros(batch_size).requires_grad_().cuda()
    # dy = torch.zeros(batch_size).requires_grad_().cuda()
    # theta = torch.zeros(batch_size).cuda()
    # for i in range(len(predictions[0])):
    # # if predictions[:,i] == 0:
    #     dx[predictions[:, i, 0] > 0] = torch.cos(theta)[predictions[:, i, 0] > 0]
    # dy[predictions[:, i, 0] > 0] = torch.sin(theta)[predictions[:, i, 0] > 0]
    #
    # # if predictions[:,i] == 1:
    # theta[predictions[:, i, 1] > 0] = theta[predictions[:, i, 1] > 0] + dtheta
    #
    # # if predictions[:,i] == 2:
    # theta[predictions[:, i, 2] > 0] = theta[predictions[:, i, 2] > 0] - dtheta
    #
    # # if predictions[:,i] == 3:
    # theta[predictions[:, i, 3] > 0] = theta[predictions[:, i, 3] > 0] + dtheta
    # dx[predictions[:, i, 3] > 0] = torch.cos(theta)[predictions[:, i, 3] > 0]
    # dy[predictions[:, i, 3] > 0] = torch.sin(theta)[predictions[:, i, 3] > 0]
    #
    # # if predictions[:,i] == 4:
    # theta[predictions[:, i, 4] > 0] = theta[predictions[:, i, 4] > 0] - dtheta
    # dx[predictions[:, i, 4] > 0] = torch.cos(theta)[predictions[:, i, 4] > 0]
    # dy[predictions[:, i, 4] > 0] = torch.sin(theta)[predictions[:, i, 4] > 0]
    #
    # pred_points_x.append(pred_points_x[i] + dx)
    # pred_points_y.append(pred_points_y[i] + dy)
    # dx = torch.zeros(batch_size).requires_grad_().cuda()
    # dy = torch.zeros(batch_size).requires_grad_().cuda()