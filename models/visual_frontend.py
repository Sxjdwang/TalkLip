"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetLayer(nn.Module):

    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->
    """

    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1,1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        return


    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch



class ResNet(nn.Module):

    """
    An 18-layer ResNet architecture.
    """

    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4,4), stride=(1,1))#4
        return


    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch



class VisualFrontend(nn.Module):

    """
    A visual feature extraction module. Generates a 512-dim feature vector per video frame.
    Architecture: A 3D convolution block followed by an 18-layer ResNet.
    """

    def __init__(self, minlen=None):
        super(VisualFrontend, self).__init__()
        self.frontend3D = nn.Sequential(
                            nn.Conv3d(1, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False),
                            nn.BatchNorm2d(64, momentum=0.01, eps=0.001),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
                        )
        self.resnet = ResNet()
        self.minlen = minlen
        return

    def outpadding(self, output, inputLen, lenReq):

        """
        output:a list of (T*512) T is changable
        inputLen: 1d tensor
        lenReq: 1d tensor
        """

        leftpadding = torch.floor((lenReq - inputLen).float()/2).int()
        rightpadding = torch.max(lenReq) - leftpadding - inputLen
        for i in range(len(output)):
            output[i] = F.pad(output[i], (0, 0, leftpadding[i], rightpadding[i])).unsqueeze(dim=0)
        return torch.cat(output, dim=0)


    def forward(self, inputBatch, inputLen, lenReq):
        # data = []
        batchsize = inputBatch.shape[0]

        batch = self.frontend3D[0](inputBatch).transpose(1, 2)
        # data.append(batch.detach().clone())
        batch2d = torch.cat([batch[i, :inputLen[i]] for i in range(inputLen.shape[0])], dim=0)
        batch = self.frontend3D[1:](batch2d)

        # data.append(batch.detach().clone())

        outputResnet = self.resnet(batch).squeeze(dim=3).squeeze(dim=2)

        paddingList = outputResnet.split(inputLen.cpu().detach().numpy().tolist(), dim=0)
        outputBatch = self.outpadding(list(paddingList), inputLen, lenReq)

        assert batchsize == outputBatch.shape[0]
        return outputBatch, lenReq#, data
