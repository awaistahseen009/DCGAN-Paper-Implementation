import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from models import Discriminator , Generator 
from models.init_weights import initialize_weights
from torch.utils.data import DataLoader , Dataset
from torchvision.utils import make_grid
from utils import save_model , load_model
# Device Agnostic Code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
BATCH_SIZE = 128 # Defined in the paper
LEARNING_RATE = 2e-4 # Defined in the paper
IMG_DIM = 64 # shape if 64 x 64 
LATENT_DIM = 100 # Z = 100
INPUT_CHANNELS = 3
EPOCHS = 20
IMG_TO_SHOW = 32
CKPT_EPOCHS = 5
# Creating the summary writers for fake and real
real_writer = SummaryWriter("runs/celeba/real")
fake_writer = SummaryWriter("runs/celeba/fake")


# DataSet and DataLoaders
transforms = transforms.Compose([
    transforms.Resize((IMG_DIM, IMG_DIM)), 
    transforms.ToTensor(), 
        transforms.Normalize(
            [0.5 for _ in range(INPUT_CHANNELS)], [0.5 for _ in range(INPUT_CHANNELS)]
        )
])
dataset = datasets.ImageFolder("img_align_celeba", transform = transforms)
dataloader = DataLoader(dataset , batch_size = BATCH_SIZE, shuffle = True)

# Models and weights Initialization
disc = Discriminator(INPUT_CHANNELS , IMG_DIM).to(device)
disc = initialize_weights(disc)
gen = Generator(LATENT_DIM , IMG_DIM).to(device)
gen = initialize_weights(gen)

# optimizers -> Adam is being used in paper
disc_optim = optim.Adam(disc.parameters(), lr = LEARNING_RATE , betas = (0.5 , 0.999))
gen_optim = optim.Adam(gen.parameters(), lr = LEARNING_RATE , betas = (0.5 , 0.999))

# loss/cost function or creterian
creterian = nn.BCELoss()

# Training Loop
global_step = 1
for epoch in range(EPOCHS):
    if (epoch % CKPT_EPOCHS == 0) & epoch != 0:
        disc_ckpts = {'dic_model_state_dict':disc.state_dict(), 'dic_optimizer_state_dict':disc_optim.state_dict()}
        save_model(disc_ckpts , f"./saved_model/disc/disc_model_epoch_{epoch+1}.pth.tar")
        gen_ckpts = {'gen_model_state_dict':gen.state_dict(), 'gen_optimizer_state_dict':gen_optim.state_dict()}
        save_model(gen_ckpts , f"./saved_model/gen/gen_model_epoch_{epoch+1}.pth.tar")


    for batch_num , (real_img, _) in enumerate(dataloader):
        real_img = real_img.to(device)
        latent_vector = torch.rand((BATCH_SIZE , LATENT_DIM)).to(device)
        fake_img = gen(latent_vector)

        disc_real_output = disc(real_img).reshape(-1)
        disc_fake_output = disc(fake_img.detach()).reshape(-1)
        # loss = log(D(x))  + log(1  - D(G(z)))
        disc_real_loss = creterian(disc_real_output, torch.ones_like(disc_real_output))
        disc_fake_loss = creterian(disc_fake_output, torch.zeros_like(disc_fake_output))
        disc_loss = (disc_real_loss + disc_fake_loss) / 2
        real_writer.add_scalar("Disc Train Loss", disc_loss.item(), global_step = global_step)
        disc_optim.zero_grad()
        disc_loss.backward()
        disc_optim.step()


        out = disc(fake_img).reshape(-1)
        gen_loss = creterian(out , torch.ones_like(out))
        fake_writer.add_scalar("Gen Train Loss", gen_loss.item(), global_step = global_step)
        gen_optim.zero_grad()
        gen_loss.backward(retain_graph = True)
        gen_optim.step()    

        if batch_num%100 == 0:
            print(f"Epoch: {epoch+1}/{EPOCHS},Batch: {batch_num}/{len(dataloader)}, Disc Loss: {disc_loss.item()}, 'Gen Loss: {gen_loss.item()}")

            with torch.no_grad():   
                fake_images = gen(latent_vector)
                real_image_grid = make_grid(real_img[:IMG_TO_SHOW], normalize = True)
                fake_image_grid = make_grid(fake_images[:IMG_TO_SHOW], normalize = True)
                real_writer.add_image("Real Images", real_image_grid, global_step = global_step)
                fake_writer.add_image("Fake Images", fake_image_grid, global_step = global_step)
            global_step += 1








