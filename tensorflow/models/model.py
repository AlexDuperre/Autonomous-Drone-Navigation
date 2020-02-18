from torchvision.models import densenet201
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()

        ############################################
        # DenseNet section
        self.densenet = densenet201(pretrained=True)
        first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True)]
        first_conv_layer.extend(list(self.densenet.features))
        self.densenet.features = nn.Sequential(*first_conv_layer)
        self.densenet.classifier = nn.Linear(1920, input_dim-3)


        ############################################
        #LSTM

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_length, height, width = x[0].shape
        images = x[0].reshape(batch_size*seq_length, height, width)
        images = images.unsqueeze(dim=1)
        featuresA = self.densenet(images)
        featuresA = featuresA.reshape(batch_size, seq_length,-1)

        # Concatenate features together
        featuresB = x[1]
        Features = torch.cat([featuresA, featuresB.type(torch.float)], dim=2)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_().cuda()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_().cuda()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(Features, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out)
        # out.size() --> 100, 10
        return out