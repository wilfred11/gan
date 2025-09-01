import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.nn import BCEWithLogitsLoss
from tqdm.auto import tqdm


def train_gan(generator, discriminator, gen_optimizer, disc_optimizer, dataloader, epochs=200, device="cpu", plot_generation_freq=50, plot_loss_freq=20, num_gens=10, latent_dim=100):

    loss_func = BCEWithLogitsLoss()
    gen_losses, disc_losses = [],[]

    for epoch in tqdm(range(epochs)):
        generator_epoch_losses= []
        discriminator_epoch_losses=[]
        for images,_ in dataloader:
            batch_size=images.shape[0]
            ### Train discriminator ###
            noise= torch.randn(batch_size, latent_dim)

            generated_labels=torch.zeros(batch_size,1)
            true_labels=torch.ones(batch_size,1)
            ### generate some samples ###
            generated_images= generator(noise).detach()

            real_discriminator_pred = discriminator(images)
            gen_discriminator_pred = discriminator(generated_images)

            ### Compute loss ###

            real_loss = loss_func(real_discriminator_pred, true_labels)
            fake_loss= loss_func(gen_discriminator_pred, generated_labels)
            disc_loss = (real_loss+fake_loss)/2
            discriminator_epoch_losses.append(disc_loss.item())

            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()

            ### train generator ###

            noise= torch.randn(batch_size, latent_dim, device=device)
            generated_images = generator(noise)
            gen_discriminator_pred = discriminator(generated_images)

            generator_loss = loss_func(gen_discriminator_pred, true_labels)
            generator_epoch_losses.append(generator_loss.item())

            gen_optimizer.zero_grad()
            generator_loss.backward()
            gen_optimizer.step()

        generator_epoch_losses = np.mean(generator_epoch_losses)
        discriminator_epoch_losses= np.mean(discriminator_epoch_losses)

        if epoch % plot_loss_freq == 0:
            print(f"Epoch:{epoch}/{epoch} | Generator Loss: {generator_epoch_losses} | Discriminator Loss: {discriminator_epoch_losses}")

        if epoch % plot_generation_freq == 0:
            generator.eval()
            with torch.no_grad():
                noise_sample = torch.randn(num_gens, latent_dim)
                generated_images = generator(noise_sample)

                fig, ax = plt.subplots(1, num_gens, figsize=(15,5))

                for i in range(num_gens):
                    img= (generated_images[i].squeeze()+1)/2
                    ax[i].imshow(img.numpy(), cmap="gray")
                    ax[i].set_axis_off()

                plt.savefig("examples_epoch_"+ str(epoch) +".png")
            generator.train()
    return generator, discriminator, gen_losses, disc_losses
