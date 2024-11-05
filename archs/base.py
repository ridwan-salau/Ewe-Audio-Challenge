import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN Model
class MyCNN(nn.Module):
    def __init__(self, n_classes):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.adaptive_pool1 = nn.AdaptiveAvgPool1d(199)
        self.adaptive_pool2 = nn.AdaptiveAvgPool1d(199)
        self.fc_layers = nn.Sequential(
            nn.Linear(199, 199),
            nn.BatchNorm1d(199),
            nn.LeakyReLU(),
            nn.Linear(199, 25),
            nn.BatchNorm1d(25),
            nn.LeakyReLU(),
            nn.Linear(25, n_classes)
        )     

    def forward(self, x):
        x = torch.unsqueeze(x, axis=0)
        x = torch.permute(x, (1, 0, 2))
        x = self.conv1(x)
        x = self.adaptive_pool1(x)
        x = self.adaptive_pool2(x)
        x = torch.squeeze(x)
        x = self.fc_layers(x)
        return x
    
# Weight initialization
def init_weights(n_classes):
    model = MyCNN(n_classes)
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0.0)
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=1.0, a=-2.0, b=2.0)
    return model
