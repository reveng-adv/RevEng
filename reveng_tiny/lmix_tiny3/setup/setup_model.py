import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()        
        self.main = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            )
        self.fc = nn.Sequential(
            nn.Linear(4*4*32, 512),    
            nn.ReLU(),            
            nn.Linear(512, 10),
            )
    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 4*4*32)
        x = self.fc(x)
        return x

    def get_feature(self, x):
        x = self.main(x)
        x = x.view(-1, 4*4*32)
        return x
    
class VGG_plain(nn.Module):
    def __init__(self, vgg_name, nclass, img_width=64):
        super(VGG_plain, self).__init__()
        self.img_width = img_width
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, nclass)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        width = self.img_width
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                width = width // 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=width, stride=1)]
        return nn.Sequential(*layers)

    def get_feature(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

def vgg16():
    return VGG_plain('VGG16', nclass=10)


class AdvClaMnist(nn.Module):
    def __init__(self, num_class):
        super(AdvClaMnist, self).__init__()   
        self.num_class = num_class
        self.main = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            )
        self.fc = nn.Sequential(
            nn.Linear(4*4*32, 512),    
            nn.ReLU(),            
            nn.Linear(512, self.num_class),
            )
    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 4*4*32)
        x = self.fc(x)
        return x

def AdvClaCIFAR10(num_class):
    return VGG_plain('VGG16', nclass=num_class)
