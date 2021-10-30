import torch
from aegan import Generator as G
import torchvision.utils as vutils

device = torch.device('cpu')
netG = G()
netG.load_state_dict(torch.load('trained_generator_weights.pt', map_location=device))
vec = torch.randn((32, 16))
with torch.no_grad():
    fake = netG(vec)

for i in range(32):
    vutils.save_image(fake.data[i], f'testfake.{i:02d}.png', normalize=True)
