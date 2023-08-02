'''
This module contains the models that I implemented.
'''

import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary


# a block that enables us to enter customized functions in the structure of an nn.Module
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)

# a function for flattening each feature map before Global Average Pooling
def myReshape(xb):
    return xb.view(-1,xb.shape[1], xb.shape[2]*xb.shape[3])

# A function to calculate the average of each flattened feature map
def myMean(xb):
    return torch.mean(xb, -1, keepdim = False)


# SonoNet model; default values represent SonoNet16
# This model was originally implemented in Lasagne (https://github.com/baumgach/SonoNet-weights/tree/master)
class SonoNet(nn.Module):
    def __init__(self,
        ch1 = 16,
        ch2 = 32,
        ch3 = 64,
        ch4 = 128,
        ch5 = 128,
        ada = 64,
        num_labels = 10
        ):
        
        super(SonoNet, self).__init__()
        
        self.ch1 = ch1
        self.ch2 = ch2
        self.ch3 = ch3
        self.ch4 = ch4
        self.ch5 = ch5
        self.ada = ada
        self.K = num_labels
        self.name = 'SonoNet' + str(self.ch1)
        
        # 1st Layer
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=self.ch1, kernel_size=3)
        self.bn1_1 = nn.BatchNorm2d(num_features=self.ch1)
        self.act1_1 = nn.ReLU()
        
        self.conv1_2 = nn.Conv2d(in_channels=self.ch1, out_channels=self.ch1, kernel_size=3)
        self.bn1_2 = nn.BatchNorm2d(num_features=self.ch1)
        self.act1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2nd Layer
        self.conv2_1 = nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=3)
        self.bn2_1 = nn.BatchNorm2d(num_features=self.ch2)
        self.act2_1 = nn.ReLU()
        
        self.conv2_2 = nn.Conv2d(in_channels=self.ch2, out_channels=self.ch2, kernel_size=3)
        self.bn2_2 = nn.BatchNorm2d(num_features=self.ch2)
        self.act2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3rd Layer
        self.conv3_1 = nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=3)
        self.bn3_1 = nn.BatchNorm2d(num_features=self.ch3)
        self.act3_1 = nn.ReLU()
        
        self.conv3_2 = nn.Conv2d(in_channels=self.ch3, out_channels=self.ch3, kernel_size=3)
        self.bn3_2 = nn.BatchNorm2d(num_features=self.ch3)
        self.act3_2 = nn.ReLU()
        
        self.conv3_3 = nn.Conv2d(in_channels=self.ch3, out_channels=self.ch3, kernel_size=3)
        self.bn3_3 = nn.BatchNorm2d(num_features=self.ch3)
        self.act3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 4th Layer
        self.conv4_1 = nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=3)
        self.bn4_1 = nn.BatchNorm2d(num_features=self.ch4)
        self.act4_1 = nn.ReLU()
        
        self.conv4_2 = nn.Conv2d(in_channels=self.ch4, out_channels=self.ch4, kernel_size=3)
        self.bn4_2 = nn.BatchNorm2d(num_features=self.ch4)
        self.act4_2 = nn.ReLU()
        
        self.conv4_3 = nn.Conv2d(in_channels=self.ch4, out_channels=self.ch4, kernel_size=3)
        self.bn4_3 = nn.BatchNorm2d(num_features=self.ch4)
        self.act4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
         # 5th Layer
        self.conv5_1 = nn.Conv2d(in_channels=self.ch4, out_channels=self.ch5, kernel_size=3)
        self.bn5_1 = nn.BatchNorm2d(num_features=self.ch5)
        self.act5_1 = nn.ReLU()
        
        self.conv5_2 = nn.Conv2d(in_channels=self.ch5, out_channels=self.ch5, kernel_size=3)
        self.bn5_2 = nn.BatchNorm2d(num_features=self.ch5)
        self.act5_2 = nn.ReLU()
        
        self.conv5_3 = nn.Conv2d(in_channels=self.ch5, out_channels=self.ch5, kernel_size=3)
        self.bn5_3 = nn.BatchNorm2d(num_features=self.ch5)
        self.act5_3 = nn.ReLU()
        
        # Adaptation Layer
        self.conva_1 = nn.Conv2d(in_channels=self.ch5, out_channels=self.ada, kernel_size=(1,1))
        self.bna_1 = nn.BatchNorm2d(num_features=self.ada)
        self.acta_1 = nn.ReLU()
        
        self.conva_2 = nn.Conv2d(in_channels=self.ada, out_channels=self.K, kernel_size=(1,1))
        self.bna_2 = nn.BatchNorm2d(num_features=self.K)
        
        # Global Average Pooling
        self.reshape = Lambda(myReshape)
        self.average = Lambda(myMean)
        
        # Classification
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, xb):
        xb = self.conv1_1(xb)
        xb = self.bn1_1(xb)
        xb = self.act1_1(xb)
        
        xb = self.conv1_2(xb)
        xb = self.bn1_2(xb)
        xb = self.act1_2(xb)
        xb = self.pool1(xb)
        
        xb = self.conv2_1(xb)
        xb = self.bn2_1(xb)
        xb = self.act2_1(xb)
        
        xb = self.conv2_2(xb)
        xb = self.bn2_2(xb)
        xb = self.act2_2(xb)
        xb = self.pool2(xb)
        
        xb = self.conv3_1(xb)
        xb = self.bn3_1(xb)
        xb = self.act3_1(xb)
        
        xb = self.conv3_2(xb)
        xb = self.bn3_2(xb)
        xb = self.act3_2(xb)

        xb = self.conv3_3(xb)
        xb = self.bn3_3(xb)
        xb = self.act3_3(xb)
        xb = self.pool3(xb)

        xb = self.conv4_1(xb)
        xb = self.bn4_1(xb)
        xb = self.act4_1(xb)

        xb = self.conv4_2(xb)
        xb = self.bn4_2(xb)
        xb = self.act4_2(xb)

        xb = self.conv4_3(xb)
        xb = self.bn4_3(xb)
        xb = self.act4_3(xb)
        xb = self.pool4(xb)

        xb = self.conv5_1(xb)
        xb = self.bn5_1(xb)
        xb = self.act5_1(xb)

        xb = self.conv5_2(xb)
        xb = self.bn5_2(xb)
        xb = self.act5_2(xb)

        xb = self.conv5_3(xb)
        xb = self.bn5_3(xb)
        xb = self.act5_3(xb)

        xb = self.conva_1(xb)
        xb = self.bna_1(xb)
        xb = self.acta_1(xb)

        xb = self.conva_2(xb)
        xb = self.bna_2(xb)

        xb = self.reshape(xb)
        xb = self.average(xb)

        xb = self.softmax(xb)
        
        return xb


#  Siren, from https://github.com/vsitzmann/siren
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, self.out_features, bias = bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1/self.in_features, 1/self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6/self.in_features)/self.omega_0, np.sqrt(6/self.in_features)/self.omega_0)

    def forward(self, xb):
        return torch.sin(self.omega_0 * self.linear(xb))

class Siren(nn.Module):
    def __init__(
        self, in_features, hidden_features, hidden_layers, out_features, last_linear=False,
        first_omega_0=30, hidden_omega_0 = 30
    ):

        super().__init__()

        self.net = list()
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0))

        if last_linear:
            final_layer = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_layer.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_layer)
        else:
            self.net.append(SineLayer(hidden_features, out_features, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, xb):
        xb = xb.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        yb = self.net(xb)
        return yb, xb
    

# Latent Modulated Siren, from Dupont et al. https://arxiv.org/abs/2201.12204
class ModulatedSineLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_modulations = None,
        bias=True,
        is_first=False,
        omega_0=30,
        device = 'cpu'
    ):

        super().__init__()

        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(self.in_features, self.out_features, bias = bias).to(device)
        self.init_weights()

        self.num_modulations = num_modulations
        if self.num_modulations != None:
            assert type(num_modulations) == int, 'TypeError: num_modulations is either None or int.'
            self.modulator = nn.Linear(self.num_modulations, self.out_features).to(device)

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1/self.in_features, 1/self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6/self.in_features)/self.omega_0, np.sqrt(6/self.in_features)/self.omega_0)

    def forward(self, xb, modulation=None):
        xb = self.linear(xb)

        if modulation != None and self.num_modulations != None:
            xb += self.modulator(modulation)

        return torch.sin(self.omega_0 * xb)
    

class ModulatedSiren(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        num_modulations,
        out_features,
        last_linear = False,
        first_omega_0 = 50,
        hidden_omega_0 = 50,
        device = 'cpu'
    ):

        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features # list of hidden features
        self.num_modulations = num_modulations # number of modulations (None or some int)
        self.out_features = out_features
        self.last_linear = last_linear
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        self.device = device

        self.net = list()

        self.net.append(
            ModulatedSineLayer(
                self.in_features,
                self.hidden_features[0],
                num_modulations=self.num_modulations,
                is_first=True,
                omega_0=self.first_omega_0,
                device = self.device
            )
        )

        for i in range(len(self.hidden_features)-1):
            self.net.append(
                ModulatedSineLayer(
                    self.hidden_features[i],
                    self.hidden_features[i+1],
                    num_modulations=self.num_modulations,
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                    device = self.device
                )
            )

        if self.last_linear:
            final_layer = nn.Linear(hidden_features[-1], out_features).to(self.device)
            with torch.no_grad():
                final_layer.weight.uniform_(-np.sqrt(6 / self.hidden_features[-1]) / self.hidden_omega_0,
                                              np.sqrt(6 / self.hidden_features[-1]) / self.hidden_omega_0)
            self.net.append(final_layer)
        else:
            self.net.append(
                ModulatedSineLayer(
                    self.hidden_features[-1],
                    out_features,
                    num_modulations=self.num_modulations,
                    omega_0=self.hidden_omega_0,
                    device = self.device
                )
            )

        self.net = nn.Sequential(*self.net).to(device)
        self.modulation = torch.zeros(size=[self.num_modulations], requires_grad=True).to(self.device)

    def reset_modulation(self):
        self.modulation = self.modulation.detach() * 0
        self.modulation = self.modulation.to(self.device)
        self.modulation.requires_grad = True

    def forward(self, xb):
        for layer in self.net[:-1]:
            xb = layer(xb, modulation = self.modulation)
        if self.last_linear:
            xb = self.net[-1](xb)
        else:
            xb = self.net[-1](xb, modulation = self.modulation)
        return xb

if __name__ == "__main__":
    model = SonoNet()
    summary(model, (1,224,288), device='cpu')
