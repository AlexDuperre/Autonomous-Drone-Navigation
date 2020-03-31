import torch
import torch.nn as nn

class weightedLoss(nn.Module):
    def __init__(self):
        super(weightedLoss, self).__init__()
        self.crossentropy = nn.CrossEntropyLoss(weight=torch.Tensor([0.8213502735, 2.6, 2.116669019, 4.3366425512, 4.3308459881]).cuda(),reduction="none") #[0.0684208353, 0.0213502735, 0.1260713329, 0.116669019, 0.3366425512, 0.3308459881]

    def sample_weighter(self, depth):
        batch, frame_nb, _, _ = depth.shape
        medians, _ = depth.view(batch, frame_nb, -1).median(dim=2)
        cond = medians <= 0.14
        weights = torch.zeros(medians.shape).cuda()
        weights[cond] = torch.ones(medians.shape)[cond].cuda() - torch.div(medians[cond],0.2)
        weights += torch.ones(medians.shape).cuda()
        weights = weights**4
        weights = weights / weights.sum()
        return weights.cuda()


    def forward(self, outputs, targets, depth):
        weights = self.sample_weighter(depth).view(-1)

        losses = self.crossentropy(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
        loss = (losses * weights).sum()

        return loss
