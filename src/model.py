import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models.resnet import ResNet, BasicBlock

class EarlyRN18(ResNet):
    def __init__(self, pretrained, freeze, label_len):
        super().__init__(BasicBlock, [2, 2, 2, 2])
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    
        if pretrained:
            print("loading imagenet weights")
        self.load_state_dict(models.resnet18(weights=weights).state_dict())

        if freeze:
            print("freezing encoder")
            for p in self.parameters():
                p.requires_grad = False

        num_ftrs = self.fc.in_features
        self.late_fc = nn.Linear(num_ftrs, label_len)
        self.early_fc = nn.Linear(200704, label_len) # 100352 for exit after 2nd res block, 2x that for after 1st block


    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        early_out = torch.flatten(x, 1)
        early_out = self.early_fc(early_out)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.late_fc(x)

        return x, early_out

class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 100, 3, stride=1, padding=1) # num_params = out_c * (in_c * kernel * kernel + 1 bias)
        self.conv2 = nn.Conv2d(100, 100, 3, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(6*6*100, 10)
        self.early_fc = nn.Linear(14 * 14 * 100, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)

        early_out = torch.flatten(x, 1)
        early_out = self.early_fc(early_out)

        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, early_out

def load_model(model, pretrained, freeze, label_len):
    if model == 'EarlyRN18':
        return EarlyRN18(pretrained=pretrained, freeze=freeze, label_len=label_len)
    elif model == 'CNN':
        return CNN()