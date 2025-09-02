from torch import nn

class MNISTDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminator=nn.Sequential(
            nn.Linear(784,1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )

    def forward(self,x):
        batch_size = x.shape[0]
        #print(x.shape)
        x=x.reshape(batch_size,-1)
        #print(x.shape)
        return self.discriminator(x)


class MNISTGenerator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.generator =  nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, noise):
        bath_size=noise.shape[0]
        generated = self.generator(noise)
        generated = generated.reshape(bath_size, 1,28,28)
        return generated


