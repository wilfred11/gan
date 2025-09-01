## GAN

I have used this [video](https://www.youtube.com/watch?v=R9VOZnKEBE0&t=591s) to find out how a gan works.

The generative adverserial network (GAN) is a system composed of two networks, a generator and a discriminator. The generator is supposed to generate fake data, while the discriminator is supposed to decide whether or not the generated data is real. The only reason the discriminator exists is to allow the generator to get better at generating fake data resembling the real data. So the discriminator is trying to reduce the loss when discriminating, and the generator would try to trick the discriminator into thinking the generated data is real.

### Discriminator

The discriminator layer ends up with a linear layer returning 1 feature, this feature should eventually have a value between 0 or a 1, the probability that it's fake or real. A zero should represent the fact the presented data is fake or real. 
The forward function reshapes (flattens) the input of batch  size times (28*28) into a shape of batch size times 784, which is what the discriminator layer expects.

```
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
        x=x.reshape(batch_size,-1)
        return self.discriminator(x)
```

### Generator

The generator looks like below. During training it is being fed with a two dimensional tensor (in this case a 2D tensor with dimensions batch size and 100). 
The generator function converts the 100 numbers of noise into a flattened tensor of 784. Which is the same size the discriminator expects. The Tanh() function will scale the values of to values between -1 and 1.

```
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
```

### Updating the weights

The generator and the discriminator are being trained one at a time. As both networks have different objectives, training them at the same time wouldn't work.

The discriminator will be fed with real and fake data, and will be trained to minimize the BCELoss. The BCELoss is a criterion that measures the Binary Cross Entropy between the target and the input probabilities. The difference should be as small as possible. This minimizing process should only occur for the discriminator. As the generator will want to maximize the chance of tricking the discriminator.

To achieve this the fake (generated) data should be detached after generating it. Detaching data will make the data forget its origin, so it can be fed into the discriminator as fake data. The other part of the data fed into the discriminator will be real data. Both real and fake data will be accompanied by labels. So when backpropagating the BCELoss, the discriminator will get better at discriminating between real and fake data.

To train the generator, the BCELoss should get propagated through the discriminator and the generator, but only the generator should update its weights. 

It is not completely necessary that this process should follow the exact order in which it is described here. But it is important to have two different stages one in which updating the generator is the target, and another one in which updating the discriminator is the target. 

### Training the discriminator

Training the discriminator looks like below.

```
generated_images= generator(noise).detach()

real_discriminator_pred = discriminator(images)
gen_discriminator_pred = discriminator(generated_images)

### Compute loss ###

real_loss = loss_func(real_discriminator_pred, true_labels)
fake_loss= loss_func(gen_discriminator_pred, generated_labels)
disc_loss = (real_loss+fake_loss)/2

disc_optimizer.zero_grad()
disc_loss.backward()
disc_optimizer.step()
```

The real images are fed through the discriminator and the fake (detached) images are being fed through discriminator. The loss is being backpropagated through the discriminator, but not through the generator. Also visible is the optimizer that is only used in this process. 

### Training the generator

Training the generator looks like below. 

```
noise= torch.randn(batch_size, latent_dim, device=device)
generated_images = generator(noise)
gen_discriminator_pred = discriminator(generated_images)

generator_loss = loss_func(gen_discriminator_pred, true_labels)
generator_epoch_losses.append(generator_loss.item())

gen_optimizer.zero_grad()
generator_loss.backward()
gen_optimizer.step()
```

The generated images are being fed into the discriminator, after which the chance of an image being real is being compared to the true value of the image. The loss that is thus generated by the discriminator is backpropagated through the generator to train it. Also visible in this part is the optimizer only used in this process. 

