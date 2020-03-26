from torchvision.models import densenet201
import torch
import torch.nn as nn
from GPUtil import showUtilization as gpu_usage

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, Pretrained=False):
        super(LSTMModel, self).__init__()

        ############################################
        # DenseNet section
        # self.densenet = densenet201(pretrained=True)
        # first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True)]
        # first_conv_layer.extend(list(self.densenet.features))
        # self.densenet.features = nn.Sequential(*first_conv_layer)
        # self.densenet.classifier = nn.Linear(1920, input_dim-3)

        self.densenet = ResCNN()
        if Pretrained:
            state_dict = torch.load('ResCNN_model.pt')
            self.densenet.load_state_dict(state_dict)


        ############################################
        #LSTM

        # self.fc1 = nn.Linear(6,100)
        # self.fc2 = nn.Linear(100,1000)

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc1 = nn.Linear(hidden_dim, 250)
        # self.bn1 = nn.BatchNorm1d(250)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(250, output_dim)
        self.dropout = nn.Dropout()

        self.fc = nn.Sequential(
            self.fc1,
            # self.bn1,
            self.dropout,
            self.fc2,

        )

    def forward(self, x, hn, cn):
        batch_size, seq_length, height, width = x[0].shape

        images = x[0].reshape(batch_size*seq_length, height, width)
        images = images.unsqueeze(dim=1)
        featuresA = self.densenet(images)
        featuresA = featuresA.reshape(batch_size, seq_length,-1)

        # Concatenate features together
        featuresB = x[1]
        Features = featuresA #torch.cat([featuresA, featuresB.type(torch.float)], dim=2)

        # Representation of the input
        # out = self.fc1(Features)
        # out = self.fc2(out)

        # Initialize hidden state with zeros
        # h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_().cuda()
        #
        # # Initialize cell state
        # c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_().cuda()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(Features, (hn, cn))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out)

        # out.size() --> 100, 10
        return out, (hn, cn)


class ResCNN(nn.Module):
    def __init__(self):
        super(ResCNN, self).__init__()

        self.conv1 = nn.Conv2d(1,16, 5)
        self.batchNorm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(2)


        self.conv2 = nn.Conv2d(16, 32,5)
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()


        self.conv3 = nn.Conv2d(32, 32, 3)
        self.batchNorm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2)


        self.fc1 = nn.Linear(23040, 5000)
        self.relu4 = nn.ReLU()
        self.fc2 =  nn.Linear(5000, 450)
        # self.fc3 = nn.Linear(150, 6)

        # Define layers
        self.layer1 = nn.Sequential(
            self.conv1,
            self.batchNorm1,
            self.relu1,
            self.avgpool1
        )

        self.layer2 = nn.Sequential(
            self.conv2,
            self.batchNorm2,
            self.relu2
        )

        self.layer3 = nn.Sequential(
            self.conv3,
            self.batchNorm3,
            self.relu3,
            self.avgpool2
        )

        self.layer4 = nn.Sequential(
            Squeeze(),
            self.fc1,
            self.relu4,
            self.fc2
        )

    def forward(self,x):

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        return out4

class Squeeze(nn.Module):
    def __init__(self, *args):
        super(Squeeze, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Unsqueeze(nn.Module):
    def __init__(self, *args):
        super(Unsqueeze, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.shape[0], 32, 20, 36)


class ResCNNDecoder(nn.Module):
    def __init__(self):
        super(ResCNNDecoder, self).__init__()


        self.fc1 = nn.Linear(450, 5000)
        self.relu1 = nn.ReLU()
        self.fc2 =  nn.Linear(5000, 23040)
        self.relu2 = nn.ReLU()


        self.conv1 = nn.Conv2d(32, 32, 3, 1, 3)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.upsampling1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv2 = nn.Conv2d(32, 16, 5, 1, 1)
        self.batchNorm2 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()

        self.conv3 = nn.Conv2d(16, 1, 5, 1, 3)
        self.batchNorm3 = nn.BatchNorm2d(1)
        self.relu5 = nn.ReLU()
        self.upsampling3 = nn.UpsamplingBilinear2d(scale_factor=2)


        # Define layers
        self.layer1 = nn.Sequential(
            self.fc1,
            self.relu1,
            self.fc2,
            self.relu2,
            Unsqueeze()
        )

        self.layer2 = nn.Sequential(
            self.conv1,
            self.batchNorm1,
            self.relu3,
            self.upsampling1
        )

        self.layer3 = nn.Sequential(
            self.conv2,
            self.batchNorm2,
            self.relu4
        )

        self.layer4 = nn.Sequential(
            self.conv3,
            self.batchNorm3,
            self.relu5,
            self.upsampling3
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        return out4


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.Encoder = ResCNN()
        self.Decoder = ResCNNDecoder()

    def forward(self, x):
        out = self.Encoder(x)
        x = self.Decoder(out)

        return x
