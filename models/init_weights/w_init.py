import torch.nn as nn
def initialize_weights(model):
    for module in model.modules():
        if isinstance(module , (nn.Conv2d, nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data , 0.0 , 0.02)
    return model
