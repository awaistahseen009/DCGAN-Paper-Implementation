import torch
import torch.nn as nn
'''
GUIDELINES FROM THE PAPER 
Architecture guidelines for stable Deep Convolutional GANs
• Replace any pooling layers with strided convolutions (discriminator) and fractional-strided
convolutions (generator).
• Use batchnorm in both the generator and the discriminator.
• Remove fully connected hidden layers for deeper architectures.
• Use ReLU activation in generator for all layers except for the output, which uses Tanh.
• Use LeakyReLU activation in the discriminator for all layers
'''
class Discriminator(nn.Module):
    def __init__(self,input_channels , image_dim):
        super().__init__()
        self.disc = nn.Sequential(
            self.block(in_channels=input_channels , out_channels=image_dim), # 64 , 32 x 32  
            self.block(in_channels= image_dim, out_channels=image_dim * 2,batch_norm=True), # 128 , 16 x 16
            self.block(in_channels=image_dim * 2 , out_channels=image_dim * 4, batch_norm=True ), # 256 , 8 X 8
            self.block(in_channels=image_dim * 4 , out_channels=image_dim * 8 ,  batch_norm=True), # 512, 4 x 4 
            self.block(in_channels=image_dim * 8 , out_channels=1 ,padding = 0, batch_norm=False, bias = False),
            nn.AdaptiveAvgPool2d(output_size=(1 ,1)),
            nn.Sigmoid()
        )
    def block(self, in_channels , out_channels , kernel_size = 4 , stride = 2, padding = 1,bias = True ,batch_norm = False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size , stride=stride, padding=padding, bias=True if bias else None), 
                nn.LeakyReLU(0.2)
                ]
        if batch_norm:
            layers.insert(1,nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)
    
    def forward(self , X):
        return self.disc(X)
