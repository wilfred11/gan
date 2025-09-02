## GAN

I have used this [video](https://www.youtube.com/watch?v=R9VOZnKEBE0&t=591s) to find out how a gan works.

The generative adverserial network (GAN) is a system composed of two networks, a generator and a discriminator. The generator is supposed to generate fake data, while the discriminator is supposed to decide whether or not the generated data is real. The only reason the discriminator exists is to allow the generator to get better at generating fake data resembling the real data. So the discriminator is trying to reduce the loss when discriminating, and the generator would try to trick the discriminator into thinking the generated data is real. In this case the MNIST dataset containing images of digits (28*28 pixels), will be used. The gan will generate random images, so it is not known on beforehand which number will be generated.

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

### Generated examples

Some generated digits before doing any training.

<img width="1500" height="500" alt="examples_epoch_0" src="https://github.com/user-attachments/assets/5c8e26ad-a125-4581-9e11-4aa321c7d933" />

After 50 epochs of 64 images each.

<img width="1500" height="500" alt="examples_epoch_50" src="https://github.com/user-attachments/assets/682deda2-1221-4313-afa7-5157d7ee235d" />

After 100 epochs.

<img width="1500" height="500" alt="examples_epoch_100" src="https://github.com/user-attachments/assets/d6af9c9e-162c-4daa-a51a-4b9d801f3f43" />

After 150 epochs.

<img width="1500" height="500" alt="examples_epoch_150" src="https://github.com/user-attachments/assets/28973ec3-11f6-4999-a272-907f232708d0" />

### Some remarks on GAN

#### Dynamic equilibrum

GAN training involves a dynamic equilibrium between the generator and discriminator networks. As one network improves, the other must adapt to maintain the balance. This dynamic nature makes it challenging to determine when the training process has converged, as the networks may continue to evolve and improve even after the loss values stabilize.

#### Non-convex optimization

GANs involve a non-convex optimization problem. The objective function being optimized is non-convex due to the adversarial nature of the training process. This means that the loss landscape contains multiple local minima, making it difficult to find the global optimum. The non-convexity of the objective function can lead to convergence issues and make it challenging to train GANs effectively.

#### Mode colapse

Mode collapse in GANs refers to a scenario where the generator produces a limited variety of samples, often focusing on a few modes of data distribution while ignoring large parts of the data distribution. This phenomenon leads to a lack of diversity in the generated samples, resulting in poor quality and unrealistic outputs.

#### Training instability 

GAN training is highly sensitive to hyperparameters, architecture choices, and initialization[3]. Small changes in these factors can lead to unstable training dynamics, such as oscillations or divergence. Additionally, the discriminator and generator networks may become unbalanced during training, leading to one network overpowering the other.

#### Mitigation

The above challenges can be mitigated by employing diversity-promoting techniques[4], such as adding noise to the input of the generator, using mini-batch discrimination, using alternative loss functions, incorporating regularization terms in the loss function, or designing more complex architectures that can better capture the diversity of the data distribution. 

## The conditional GAN

This GAN makes it possible to generate a labeled image, One can choose the digit to generate.

The main difference between a normal gan and the condiitonal gan is that a label embedding gets concatenated to the noise when generating fake noise. And the discriminator tries to tell from the noise with embedded label whether it is real or fake. Of course the discriminator is fed with real data and real label too.  

### CGAN Generator

```
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
```

### CDiscriminator

```
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
```

### Training the CGAN Discriminator

Generating fake data. The rand_digits are the digits by which the noise will be represented.

The discriminator is used to generate some predictions, as to whether the presented image tensor is true or false.

```
noise= torch.randn(batch_size, latent_dim)
rand_digits = torch.randint(0, num_classes, size=(batch_size, ))
generated_labels=torch.zeros(batch_size,1)

generated_images= generator(noise, rand_digits).detach()

real_discriminator_pred = discriminator(images, true_digits)
gen_discriminator_pred = discriminator(generated_images, rand_digits)

real_loss = loss_func(real_discriminator_pred, true_labels)
fake_loss= loss_func(gen_discriminator_pred, generated_labels)
disc_loss = (real_loss+fake_loss)/2

disc_optimizer.zero_grad()
disc_loss.backward()
disc_optimizer.step()
```

### Training the CGAN Generator

Training the conditional generator is almost identical to the simple generator. In this case the discriminator predictions should move towards the true_labels, meaning the discriminator mistakes the generated images for real. That's why the loss_func is fed with discriminator predictions and the true_labels.

```
 rand_digits= torch.randint(0,num_classes, size=(batch_size,))
 noise= torch.randn(batch_size, latent_dim, device=device)
 generated_images = generator(noise, rand_digits)
 gen_discriminator_pred = discriminator(generated_images, rand_digits)

 generator_loss = loss_func(gen_discriminator_pred, true_labels)

 gen_optimizer.zero_grad()
 generator_loss.backward()
 gen_optimizer.step()
```
### Generated examples

Some generated digits before doing any training.

<img width="1500" height="500" alt="examples_epoch_0" src="https://github.com/user-attachments/assets/5c8e26ad-a125-4581-9e11-4aa321c7d933" />

After 50 epochs of 64 images each.

<img width="1500" height="500" alt="examples_epoch_50" src="https://github.com/user-attachments/assets/682deda2-1221-4313-afa7-5157d7ee235d" />

After 100 epochs.

<img width="1500" height="500" alt="examples_epoch_100" src="https://github.com/user-attachments/assets/d6af9c9e-162c-4daa-a51a-4b9d801f3f43" />

After 150 epochs.

<img width="1500" height="500" alt="examples_epoch_150" src="https://github.com/user-attachments/assets/28973ec3-11f6-4999-a272-907f232708d0" />


