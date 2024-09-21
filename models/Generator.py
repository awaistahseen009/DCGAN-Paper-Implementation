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
class Generator(nn.Module):
    def __init__(self , latent_dim , image_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.Unflatten(dim=1, unflattened_size=(latent_dim, 1, 1)),
            self.block_(latent_dim , out_channels=image_dim * 16 , kernel_size=4 , stride = 1, padding = 0 , batch_norm=True),#1024 , 4 x 4
            self.block_(image_dim * 16 , out_channels=image_dim * 8  , kernel_size=4 , stride=2 , padding = 1 , batch_norm=True), #512 , 8 x 8
            self.block_(in_channels=image_dim * 8 , out_channels=image_dim * 4 , kernel_size=4 , stride = 2 , padding=1 , batch_norm=True) ,  #256 , 16 x 16
            self.block_(in_channels= image_dim * 4, out_channels=image_dim * 2 , kernel_size=4 , stride=2 , padding=1 , batch_norm=True), #128 , 32 x 32 
            self.block_(image_dim * 2, out_channels=3 , kernel_size= 4, stride = 2 , padding = 1 , bias = False, activation="tanh")

        )

    def block_(self , in_channels , out_channels, kernel_size , stride , padding, batch_norm = False, bias = True, activation = "relu"):
        layers = [
            nn.ConvTranspose2d(in_channels=in_channels , out_channels=out_channels , 
            kernel_size=kernel_size,stride=stride, padding=padding , bias= True if bias else None), 
        ]
        if batch_norm:
            layers.insert(1 , nn.BatchNorm2d(out_channels))
        if activation == "tanh":
            layers.insert(2 , nn.Tanh())
        else:
            layers.insert(2 , nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self , latent_vector):
        return self.gen(latent_vector)


        
        