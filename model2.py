import torch
from torch import nn

class CDiscriminator(nn.Module):
    def __init__(self, num_classes=10, embedding_dim=16):
        super().__init__()
        self.embeddings = nn.Embedding(num_classes, embedding_dim)
        self.discriminator = nn.Sequential(
            nn.Linear(784 +embedding_dim, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1)
        )
    def forward(self, x, labels):
        batch_size = x.shape[0]
        embeddings= self.embeddings(labels)
        x=x.reshape(batch_size, -1)
        x=torch.cat([x,embeddings], dim=1)
        return self.discriminator(x)


class CGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10,embedding_dim=16):
        super().__init__()
        self.embeddings = nn.Embedding(num_classes, embedding_dim)
        self.generator = nn.Sequential(
            nn.Linear(latent_dim+embedding_dim, 256),
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

    def forward(self, noise, labels):
        bath_size = noise.shape[0]
        embeddings=self.embeddings(labels)
        noise = torch.cat([noise, embeddings], dim=-1)
        generated = self.generator(noise)
        generated = generated.reshape(bath_size, 1, 28, 28)
        return generated
