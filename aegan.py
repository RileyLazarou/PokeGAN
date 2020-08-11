import os
import json
import sys

import torch
from torch import nn
from torch import optim
import torchvision as tv
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_dim=8):
        """A generator for mapping a latent space to a sample space.

        The sample space for this generator is single-channel, 28x28 images
        with pixel intensity ranging from -1 to +1.

        Args:
            latent_dim (int): latent dimension ("noise vector")
            batchnorm (bool): Whether or not to use batch normalization
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self._init_modules()

    def build_colourspace(self, input_dim, output_dim):
        colourspace = nn.Sequential(
            nn.Linear(
                input_dim,
                128,
                bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(
                128,
                64,
                bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Linear(
                64,
                output_dim,
                bias=True),
            nn.Tanh(),
            )
        return colourspace

    def _init_modules(self):
        """Initialize the modules."""
        projection_widths = [8, 8, 8, 8, 8, 8, 8]
        self.projection_dim = sum(projection_widths) + self.latent_dim // 2
        self.leaky_relu = nn.LeakyReLU()
        self.projections = nn.ModuleList()  # (64,)
        running_sum = self.latent_dim // 2
        for i in projection_widths:
            self.projections.append(
                nn.Sequential(
                    nn.Linear(
                        running_sum,
                        i,
                        bias=True,
                        ),
                    nn.BatchNorm1d(8),
                    nn.LeakyReLU(),
                    )
                )
            running_sum += i
        self.projection_upscaler = nn.Upsample(scale_factor=3)

        self.colourspace_r = self.build_colourspace(self.latent_dim // 2, 16)  # (16,)
        self.colourspace_g = self.build_colourspace(self.latent_dim // 2, 16)  # (16,)
        self.colourspace_b = self.build_colourspace(self.latent_dim // 2, 16)  # (16,)
        self.colourspace_upscaler = nn.Upsample(scale_factor=96)

        self.seed = nn.Sequential(
                nn.Linear(
                    self.projection_dim,
                    512*3*3,
                    bias=True),
                nn.BatchNorm1d(512*3*3),
                nn.LeakyReLU(),
                )

        self.upscaling = nn.ModuleList()
        self.conv = nn.ModuleList()

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
                nn.ZeroPad2d((1, 1, 1, 1)),
                nn.Conv2d(
                        in_channels=(512)//4,
                        out_channels=512,
                        kernel_size=3,
                        stride=1,
                        padding=0,
                        bias=True
                        ),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                ))

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.Conv2d(
                        in_channels=(512 + self.projection_dim)//4,
                        out_channels=256,
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        bias=True
                        ),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                ))

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.Conv2d(
                        in_channels=(256 + self.projection_dim)//4,
                        out_channels=256,
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        bias=True
                        ),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                ))

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.Conv2d(
                        in_channels=(256 + self.projection_dim)//4,
                        out_channels=256,
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        bias=True
                        ),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                )),

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.Conv2d(
                        in_channels=(256 + self.projection_dim)//4,
                        out_channels=64,
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        bias=True
                        ),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                ))

        self.upscaling.append(nn.Upsample(scale_factor=1))
        self.conv.append(nn.Sequential(
                nn.ZeroPad2d((2, 2, 2, 2)),
                nn.Conv2d(
                        in_channels=64,
                        out_channels=16,
                        kernel_size=5,
                        stride=1,
                        padding=0,
                        bias=True
                        ),
                nn.Softmax(dim=1),
                ))

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        shape_latent_vec = input_tensor[:, :self.latent_dim//2]
        colour_latent_vec = input_tensor[:, self.latent_dim//2:]
        last = shape_latent_vec
        for module in self.projections:
            projection = module(last)
            last = torch.cat((last, projection), -1)
        projection = last

        intermediate = self.seed(projection)
        intermediate = intermediate.view((-1, 512, 3, 3))

        projection_2d = projection.view((-1, self.projection_dim, 1, 1))
        projection_2d = self.projection_upscaler(projection_2d)

        for i, (conv, upscaling) in enumerate(zip(self.conv, self.upscaling)):
            if i + 1 != len(self.upscaling):
                if i > 0:
                    intermediate = torch.cat((intermediate, projection_2d), 1)
                intermediate = torch.nn.functional.pixel_shuffle(intermediate, 2)
            intermediate = conv(intermediate)
            projection_2d = upscaling(projection_2d)

        r_space = self.colourspace_r(colour_latent_vec)
        r_space = r_space.view((-1, 16, 1, 1))
        r_space = self.colourspace_upscaler(r_space)
        r_space = intermediate * r_space
        r_space = torch.sum(r_space, dim=1, keepdim=True)

        g_space = self.colourspace_g(colour_latent_vec)
        g_space = g_space.view((-1, 16, 1, 1))
        g_space = self.colourspace_upscaler(g_space)
        g_space = intermediate * g_space
        g_space = torch.sum(g_space, dim=1, keepdim=True)

        b_space = self.colourspace_b(colour_latent_vec)
        b_space = b_space.view((-1, 16, 1, 1))
        b_space = self.colourspace_upscaler(b_space)
        b_space = intermediate * b_space
        b_space = torch.sum(b_space, dim=1, keepdim=True)

        output = torch.cat((r_space, g_space, b_space), dim=1)

        return output


class Encoder(nn.Module):
    def __init__(self, device="cpu", latent_dim=8):
        """A discriminator for discerning real from generated images."""
        super(Encoder, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        down_channels = [3, 64, 128, 256, 512]
        self.down = nn.ModuleList()
        for i in range(len(down_channels)-1):
            self.down.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=down_channels[i],
                        out_channels=down_channels[i+1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=True,
                        ),
                    nn.BatchNorm2d(down_channels[i+1]),
                    nn.LeakyReLU(),
                    )
                )

        self.reducer = nn.Sequential(
            nn.Conv2d(
                in_channels=down_channels[-1],
                out_channels=down_channels[-2],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                ),
                nn.BatchNorm2d(down_channels[-2]),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2)
            )

        up_channels = [256, 128, 64, 64, 64]
        scale_factors = [2, 2, 2, 1]
        self.up = nn.ModuleList()
        for i in range(len(up_channels)-1):
            self.up.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=up_channels[i] + down_channels[-2-i],
                        out_channels=up_channels[i+1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        ),
                    nn.BatchNorm2d(up_channels[i+1]),
                    nn.LeakyReLU(),
                    nn.Upsample(scale_factor=scale_factors[i]),
                    )
                )

        down_again_channels = [64, 64, 64, 64, 64]
        self.down_again = nn.ModuleList()
        for i in range(len(down_again_channels)-1):
            self.down_again.append(
                nn.Conv2d(
                    in_channels=down_again_channels[i],
                    out_channels=down_again_channels[i+1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                    )
                )
            self.down_again.append(nn.BatchNorm2d(down_again_channels[i+1]))
            self.down_again.append(nn.LeakyReLU())

        self.projection = nn.Sequential(
            nn.Linear(
                512*6*6 + 64*6*6,
                256,
                bias=True,
                ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Linear(
                256,
                128,
                bias=True,
                ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(
                128,
                self.latent_dim,
                bias=True,
                ),
            )

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        rv = torch.randn(input_tensor.size(), device=self.device) * 0.02
        augmented_input = input_tensor + rv
        intermediate = augmented_input
        intermediates = [augmented_input]
        for module in self.down:
            intermediate = module(intermediate)
            intermediates.append(intermediate)
        intermediates = intermediates[:-1][::-1]

        down = intermediate.view(-1, 6*6*512)

        intermediate = self.reducer(intermediate)

        for index, module in enumerate(self.up):
            intermediate = torch.cat((intermediate, intermediates[index]), 1)
            intermediate = module(intermediate)

        for module in self.down_again:
            intermediate = module(intermediate)

        intermediate = intermediate.view(-1, 6*6*64)
        intermediate = torch.cat((down, intermediate), -1)

        projected = self.projection(intermediate)

        return projected


class DiscriminatorImage(nn.Module):
    def __init__(self, device="cpu"):
        """A discriminator for discerning real from generated images."""
        super(DiscriminatorImage, self).__init__()
        self.device = device
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        down_channels = [3, 64, 128, 256, 512]
        self.down = nn.ModuleList()
        leaky_relu = nn.LeakyReLU()
        for i in range(4):
            self.down.append(
                nn.Conv2d(
                    in_channels=down_channels[i],
                    out_channels=down_channels[i+1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                    )
                )
            self.down.append(nn.BatchNorm2d(down_channels[i+1]))
            self.down.append(leaky_relu)

        self.classifier = nn.ModuleList()
        self.width = down_channels[-1] * 6**2
        self.classifier.append(nn.Linear(self.width, 1))
        self.classifier.append(nn.Sigmoid())

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        rv = torch.randn(input_tensor.size(), device=self.device) * 0.02
        intermediate = input_tensor + rv
        for module in self.down:
            intermediate = module(intermediate)
            rv = torch.randn(intermediate.size(), device=self.device) * 0.02 + 1
            intermediate += rv

        intermediate = intermediate.view(-1, self.width)

        for module in self.classifier:
            intermediate = module(intermediate)

        return intermediate


class DiscriminatorLatent(nn.Module):
    def __init__(self, latent_dim=8,):
        """A discriminator for discerning real from generated images."""
        super(DiscriminatorLatent, self).__init__()
        self.latent_dim = latent_dim
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        self.pyramid = nn.ModuleList()
        for i in range(7):
            self.pyramid.append(
                nn.Sequential(
                    nn.Linear(
                        self.latent_dim + 8*i,
                        8,
                        bias=True,
                        ),
                    nn.BatchNorm1d(8),
                    nn.LeakyReLU(),
                    )
                )

        self.classifier = nn.ModuleList()
        self.classifier.append(nn.Linear(7*8 + self.latent_dim, 1))
        self.classifier.append(nn.Sigmoid())

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        last = input_tensor
        for module in self.pyramid:
            projection = module(last)
            last = torch.cat((last, projection), -1)
        for module in self.classifier:
            last = module(last)
        return last


class GAN():
    def __init__(self, latent_dim, noise_fn, dataloader,
                 batch_size=32, device='cpu', gen_path=None):
        """A very basic DCGAN class for generating MNIST digits

        Args:
            generator: a Ganerator network
            discriminator: A Discriminator network
            noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)
            dataloader: a pytorch dataloader for loading images
            batch_size: training batch size. Must match that of dataloader
            device: cpu or CUDA
            lr_d: learning rate for the discriminator
            lr_g: learning rate for the generator
        """
        self.latent_dim = latent_dim
        self.device = device

        self.generator = Generator(latent_dim=self.latent_dim).to(device)
        if gen_path:
            self.generator.load_state_dict(torch.load(gen_path))
        self.generator = self.generator.to(self.device)
        self.discriminator_image = DiscriminatorImage(device=self.device).to(device)

        self.noise_fn = noise_fn
        self.dataloader = dataloader
        self.batch_size = batch_size

        self.criterion = nn.BCELoss()
        self.optim_di = optim.Adam(
            self.discriminator_image.parameters(),
            lr=2e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-6,
            )
        self.optim_g = optim.Adam(
            self.generator.parameters(),
            lr=1e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-6,
            )
        self.target_ones = torch.ones((batch_size, 1), device=device)
        self.target_zeros = torch.zeros((batch_size, 1), device=device)

    def generate_samples(self, latent_vec=None, num=None):
        """Sample images from the generator.

        Images are returned as a 4D tensor of values between -1 and 1.
        Dimensions are (number, channels, height, width). Returns the tensor
        on cpu.

        Args:
            latent_vec: A pytorch latent vector or None
            num: The number of samples to generate if latent_vec is None

        If latent_vec and num are None then use self.batch_size
        random latent vectors.
        """
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        with torch.no_grad():
            samples = self.generator(latent_vec)
        samples = samples.cpu()
        return samples

    def train_step_generator(self):
        """Train the generator one step and return the loss."""
        self.generator.zero_grad()

        latent_vec = self.noise_fn(self.batch_size)
        generated = self.generator(latent_vec)
        classifications = self.discriminator_image(generated)
        loss = self.criterion(classifications, self.target_ones)
        loss.backward()
        self.optim_g.step()
        return loss.item()

    def train_step_discriminator(self, real_samples):
        """Train the discriminator one step and return the losses."""
        self.discriminator_image.zero_grad()

        # real samples
        pred_real = self.discriminator_image(real_samples)
        loss_real = self.criterion(pred_real, self.target_ones)

        # generated samples
        latent_vec = self.noise_fn(self.batch_size)
        with torch.no_grad():
            fake_samples = self.generator(latent_vec)
        pred_fake = self.discriminator_image(fake_samples)
        loss_fake = self.criterion(pred_fake, self.target_zeros)

        # combine
        loss = (loss_real + loss_fake) / 2
        loss.backward()
        self.optim_di.step()
        return loss_real.item(), loss_fake.item()

    def train_epoch(self, print_frequency=10, max_steps=0):
        """Train both networks for one epoch and return the losses.

        Args:
            print_frequency (int): print stats every `print_frequency` steps.
            max_steps (int): End epoch after `max_steps` steps, or set to 0
                             to do the full epoch.
        """
        loss_g_running, loss_d_real_running, loss_d_fake_running = 0, 0, 0
        for batch, (real_samples, _) in enumerate(self.dataloader):
            real_samples = real_samples.to(self.device)
            ldr_, ldf_ = self.train_step_discriminator(real_samples)
            loss_d_real_running += ldr_
            loss_d_fake_running += ldf_
            loss_g_running += self.train_step_generator()
            if print_frequency and (batch+1) % print_frequency == 0:
                print(f"{batch+1}/{len(self.dataloader)}:"
                      f" G={loss_g_running / (batch+1):.3f},"
                      f" Dr={loss_d_real_running / (batch+1):.3f},"
                      f" Df={loss_d_fake_running / (batch+1):.3f}",
                      end='\r',
                      flush=True)
            if max_steps and batch == max_steps:
                break
        if print_frequency:
            print()
        loss_g_running /= batch
        loss_d_real_running /= batch
        loss_d_fake_running /= batch
        return (loss_g_running, (loss_d_real_running, loss_d_fake_running))

class AEGAN():
    def __init__(self, latent_dim, noise_fn, dataloader,
                 batch_size=32, device='cpu', gen_path=None):
        """A very basic DCGAN class for generating MNIST digits

        Args:
            generator: a Ganerator network
            discriminator: A Discriminator network
            noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)
            dataloader: a pytorch dataloader for loading images
            batch_size: training batch size. Must match that of dataloader
            device: cpu or CUDA
            lr_d: learning rate for the discriminator
            lr_g: learning rate for the generator
        """
        self.latent_dim = latent_dim
        self.device = device

        self.generator = Generator(latent_dim=self.latent_dim)
        if gen_path:
            self.generator.load_state_dict(torch.load(gen_path))
        self.generator = self.generator.to(self.device)
        self.encoder = Encoder(latent_dim=self.latent_dim, device=self.device)
        self.encoder = self.encoder.to(self.device)
        self.discriminator_image = DiscriminatorImage(device=self.device).to(device)
        self.discriminator_latent = DiscriminatorLatent(
            latent_dim=self.latent_dim,
            ).to(self.device)

        self.noise_fn = noise_fn
        self.dataloader = dataloader
        self.batch_size = batch_size

        self.alphas = {
            "reconstruct_image": 10,
            "reconstruct_latent": 5,
            "discriminate_image": 0.1,
            "discriminate_latent": 0.1,
        }

        self.criterion_gen = nn.BCELoss()
        self.criterion_recon_image = nn.L1Loss()
        #self.criterion_recon_image = nn.MSELoss()
        self.criterion_recon_latent = nn.MSELoss()
        self.optim_di = optim.Adam(self.discriminator_image.parameters(),
                                   lr=2e-4, betas=(0.5, 0.999),
                                   weight_decay=1e-5)
        self.optim_dl = optim.Adam(self.discriminator_latent.parameters(),
                                   lr=2e-4, betas=(0.5, 0.999),
                                   weight_decay=1e-5)
        self.optim_g = optim.Adam(self.generator.parameters(),
                                  lr=2e-4, betas=(0.5, 0.999),
                                  weight_decay=1e-5)
        self.optim_e = optim.Adam(self.encoder.parameters(),
                                  lr=2e-4, betas=(0.5, 0.999),
                                  weight_decay=1e-5)
        self.target_ones = torch.ones((batch_size, 1), device=device)
        self.target_zeros = torch.zeros((batch_size, 1), device=device)

    def generate_samples(self, latent_vec=None, num=None):
        """Sample images from the generator.

        Images are returned as a 4D tensor of values between -1 and 1.
        Dimensions are (number, channels, height, width). Returns the tensor
        on cpu.

        Args:
            latent_vec: A pytorch latent vector or None
            num: The number of samples to generate if latent_vec is None

        If latent_vec and num are None then use self.batch_size
        random latent vectors.
        """
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        with torch.no_grad():
            samples = self.generator(latent_vec)
        samples = samples.cpu()  # move images to cpu
        return samples

    def train_step_generators(self, X):
        """Train the generator one step and return the loss."""
        self.generator.zero_grad()
        self.encoder.zero_grad()

        Z = self.noise_fn(self.batch_size)

        X_hat = self.generator(Z)
        Z_hat = self.encoder(X)
        X_tilde = self.generator(Z_hat)
        Z_tilde = self.encoder(X_hat)

        X_hat_confidence = self.discriminator_image(X_hat)
        Z_hat_confidence = self.discriminator_latent(Z_hat)
        X_tilde_confidence = self.discriminator_image(X_tilde)
        Z_tilde_confidence = self.discriminator_latent(Z_tilde)

        X_hat_loss = self.criterion_gen(X_hat_confidence, self.target_ones)
        Z_hat_loss = self.criterion_gen(Z_hat_confidence, self.target_ones)
        X_tilde_loss = self.criterion_gen(X_tilde_confidence, self.target_ones)
        Z_tilde_loss = self.criterion_gen(Z_tilde_confidence, self.target_ones)

        X_recon_loss = self.criterion_recon_image(X_tilde, X) * self.alphas["reconstruct_image"]
        Z_recon_loss = self.criterion_recon_latent(Z_tilde, Z) * self.alphas["reconstruct_latent"]

        X_loss = (X_hat_loss + X_tilde_loss) / 2 * self.alphas["discriminate_image"]
        Z_loss = (Z_hat_loss + Z_tilde_loss) / 2 * self.alphas["discriminate_latent"]
        loss = X_loss + Z_loss + X_recon_loss + Z_recon_loss
        loss.backward()
        self.optim_e.step()
        self.optim_g.step()

        return X_loss.item(), Z_loss.item(), X_recon_loss.item(), Z_recon_loss.item()

    def train_step_discriminators(self, X):
        """Train the discriminator one step and return the losses."""
        self.discriminator_image.zero_grad()
        self.discriminator_latent.zero_grad()

        Z = self.noise_fn(self.batch_size)

        with torch.no_grad():
            X_hat = self.generator(Z)
            Z_hat = self.encoder(X)
            X_tilde = self.generator(Z_hat)
            Z_tilde = self.encoder(X_hat)

        X_confidence = self.discriminator_image(X)
        X_hat_confidence = self.discriminator_image(X_hat)
        X_tilde_confidence = self.discriminator_image(X_tilde)
        Z_confidence = self.discriminator_latent(Z)
        Z_hat_confidence = self.discriminator_latent(Z_hat)
        Z_tilde_confidence = self.discriminator_latent(Z_tilde)

        X_loss = 2 * self.criterion_gen(X_confidence, self.target_ones)
        X_hat_loss = self.criterion_gen(X_hat_confidence, self.target_zeros)
        X_tilde_loss = self.criterion_gen(X_tilde_confidence, self.target_zeros)
        Z_loss = 2 * self.criterion_gen(Z_confidence, self.target_ones)
        Z_hat_loss = self.criterion_gen(Z_hat_confidence, self.target_zeros)
        Z_tilde_loss = self.criterion_gen(Z_tilde_confidence, self.target_zeros)

        loss_images = (X_loss + X_hat_loss + X_tilde_loss) / 4
        loss_latent = (Z_loss + Z_hat_loss + Z_tilde_loss) / 4
        loss = loss_images + loss_latent
        loss.backward()
        self.optim_di.step()
        self.optim_dl.step()

        return loss_images.item(), loss_latent.item()

    def train_epoch(self, print_frequency=1, max_steps=0):
        """Train both networks for one epoch and return the losses.

        Args:
            print_frequency (int): print stats every `print_frequency` steps.
            max_steps (int): End epoch after `max_steps` steps, or set to 0
                             to do the full epoch.
        """
        ldx, ldz, lgx, lgz, lrx, lrz = 0, 0, 0, 0, 0, 0
        eps = 1e-9
        for batch, (real_samples, _) in enumerate(self.dataloader):
            real_samples = real_samples.to(self.device)
            ldx_, ldz_ = self.train_step_discriminators(real_samples)
            ldx += ldx_
            ldz += ldz_
            lgx_, lgz_, lrx_, lrz_ = self.train_step_generators(real_samples)
            lgx += lgx_
            lgz += lgz_
            lrx += lrx_
            lrz += lrz_
            if print_frequency and (batch+1) % print_frequency == 0:
                print(f"{batch+1}/{len(self.dataloader)}:"
                      f" G={lgx / (eps + (batch+1) * self.alphas['discriminate_image']):.3f},"
                      f" E={lgz / (eps + (batch+1) * self.alphas['discriminate_latent']):.3f},"
                      f" Dx={ldx / (eps + (batch+1)):.3f},"
                      f" Dz={ldz / (eps + (batch+1)):.3f}",
                      f" Rx={lrx / (eps + (batch+1) * self.alphas['reconstruct_image']):.3f}",
                      f" Rz={lrz / (eps + (batch+1) * self.alphas['reconstruct_latent']):.3f}",
                      end='\r',
                      flush=True)
            if max_steps and batch == max_steps:
                break
        if print_frequency:
            print()
        lgx /= batch
        lgz /= batch
        ldx /= batch
        ldz /= batch
        lrx /= batch
        lrz /= batch
        return lgx, lgz, ldx, ldz, lrx, lrz


def save_images(GAN, vec, filename):
    images = GAN.generate_samples(vec)
    ims = tv.utils.make_grid(images, normalize=True)
    ims = ims.numpy().transpose((1,2,0))
    ims = np.array(ims*255, dtype=np.uint8)
    image = Image.fromarray(ims)
    image.save(filename)


def main():
    import matplotlib.pyplot as plt
    from time import time
    os.makedirs("results/generated", exist_ok=True)
    os.makedirs("results/reconstructed", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)

    root = os.path.join("data")
    batch_size = 32
    latent_dim = 24
    epochs = 5000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = tv.transforms.Compose([
            tv.transforms.RandomAffine(0, translate=(5/96, 5/96), fillcolor=(255,255,255)),
            tv.transforms.ColorJitter(hue=0.5),
            tv.transforms.RandomHorizontalFlip(p=0.5),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
            ])
    dataset = ImageFolder(
            root=root,
            transform=transform
            )
    dataloader = DataLoader(dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True
            )
    X = iter(dataloader)
    test_ims, _ = next(X)
    test_ims_show = tv.utils.make_grid(test_ims, normalize=True)
    test_ims_show = test_ims_show.numpy().transpose((1,2,0))
    test_ims_show = np.array(test_ims_show*255, dtype=np.uint8)
    image = Image.fromarray(test_ims_show)
    image.save("results/reconstructed/test_images.png")

    noise_fn = lambda x: torch.randn((x, latent_dim), device=device)
    test_noise = noise_fn(32)
    gen_path = None
    if len(sys.argv) > 1:
        gen_path = sys.argv[1]
        print(f"loading generator {gen_path}")
    gan = AEGAN(
        latent_dim,
        noise_fn,
        dataloader,
        device=device,
        batch_size=batch_size,
        gen_path=gen_path,
        )
    start = time()
    for i in range(epochs):
        print(f"Epoch {i+1}; Elapsed time = {int(time() - start)}s")
        gan.train_epoch(max_steps=100)
        if (i + 1) % 50 == 0:
            torch.save(gan.generator.state_dict(), f"results/checkpoints/check.{i:05d}.pt")
        save_images(gan, test_noise, f"results/generated/gen.{i:04d}.png")

        if not "encoder" in dir(gan):
            continue
        with torch.no_grad():
            reconstructed = gan.generator(gan.encoder(test_ims.cuda())).cpu()
        reconstructed = tv.utils.make_grid(reconstructed, normalize=True)
        reconstructed = reconstructed.numpy().transpose((1,2,0))
        reconstructed = np.array(reconstructed*255, dtype=np.uint8)
        reconstructed = Image.fromarray(reconstructed)
        reconstructed.save(f"results/reconstructed/gen.{i:04d}.png")

    images = gan.generate_samples()
    ims = tv.utils.make_grid(images, normalize=True)
    plt.imshow(ims.numpy().transpose((1,2,0)))
    plt.show()


if __name__ == "__main__":
    main()
