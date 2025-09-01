import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from data import get_data
from model import MNISTDiscriminator, MNISTGenerator
from train import train_gan

from torch import nn

latent_dim = 100
generator_lr=0.0001
discriminator_lr=000.1
batch_size =64




do = 4

if do==1:
    disc_model = MNISTDiscriminator()
    gen_model = MNISTGenerator(latent_dim=latent_dim)
    x=(torch.randn(2,1,28*28))
    noise = torch.randn(2,100)
    disc_model(x)

    gen= gen_model(noise)
    print(gen.shape)

if do==2:
    discriminator = MNISTDiscriminator()
    generator = MNISTGenerator(latent_dim=latent_dim)

    gen_optimizer = optim.Adam(generator.parameters(), generator_lr)
    disc_optimizer = optim.Adam(discriminator.parameters(), discriminator_lr)

    train_set, test_set= get_data()

    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    generator, discriminator, gan_losses, disc_losses = train_gan(generator, discriminator, gen_optimizer, disc_optimizer, trainloader)

if do==3:
    arr2d= np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(arr2d[-1,-1])
    print(arr2d[2,2])

if do==4:
    t=torch.randn(3,4)
    print(t)
    m = nn.Tanh()
    input = torch.randn(2)
    output = m(input)
    print(output)


