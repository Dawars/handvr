import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import functional

from pose_autoencoders.pose_loader import get_poses
from utils.render_manifold import HandRenderer

os.makedirs('./var_figures', exist_ok=True)

num_epochs = 1501
batch_size = 64
learning_rate = 1e-3
torch.manual_seed(7)

dataset = get_poses()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
poses = torch.Tensor(dataset).float().cuda()


def plot_latent(latent, epoch):
    plt.plot(*(latent.transpose()), '.', color='blue')
    plt.title("Variational Autoencoder latent space at epoch {}".format(epoch))
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.savefig("var_figures/pose_vae_latent_{0:04d}.png".format(epoch))
    plt.close()


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.l_encode1 = nn.Linear(45, 20)
        self.l_encode_std = nn.Linear(20, 2)
        self.l_encode_mean = nn.Linear(20, 2)

        self.l_decode1 = nn.Linear(2, 20)
        self.l_decode2 = nn.Linear(20, 45)

    def encoder(self, x):
        """
        :param x: Data to encode (45 values)
        :return: Encoded vector of means and, if training, standard deviations
        """
        x = self.l_encode1(x)
        x = functional.relu(x)
        if self.training:
            return self.l_encode_mean(x), self.l_encode_std(x)
        else:
            return self.l_encode_mean(x)

    def reparametize(self, means, std):
        std = torch.exp(0.5 * std)
        random_samples = torch.randn_like(std)
        return std.mul(random_samples).add_(means)

    def decoder(self, x):
        x = self.l_decode1(x)
        x = functional.relu(x)
        return self.l_decode2(x)

    def forward(self, x):
        mean, std = self.encoder(x)
        x = self.reparametize(mean, std)
        return self.decoder(x), mean, std


def loss_fcn(real_img, gen_img, mean, std):
    gen_loss = functional.mse_loss(gen_img, real_img)
    latent_loss = 0.5 * torch.sum(mean.pow(2) + std.pow(2) - torch.log(std) - 1)

    latent_loss /= batch_size
    latent_loss /= 45

    return gen_loss, latent_loss


def train():
    model = autoencoder().cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    renderer = HandRenderer(64)

    for epoch in range(num_epochs):
        for data in dataloader:
            img = data.float()
            # img = img.view(img.size(0), -1)
            img = Variable(img).cuda()

            # ===================forward=====================
            output, mean, std = model(img)
            gen_loss, latent_loss = loss_fcn(img, output, mean, std)
            loss = gen_loss + latent_loss
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================

        # plot
        if epoch % 100 == 0:
            print(
                'epoch [{}/{}], reconstruction loss:{:.4f}, KLD: {:.4f}'.format(epoch + 1, num_epochs, gen_loss.item(),
                                                                                latent_loss.item()))

            latent, _ = model.encoder(poses)
            plot_latent(latent.cpu().detach().numpy(), epoch)

            filename = "manifolds/var/var_manifold_{:04d}.png".format(epoch)
            renderer.render_manifold(model.decoder, filename, bounds=(-2, 2), steps=0.25)

    torch.save(model.state_dict(), './sim_var_autoencoder.pth')


if __name__ == '__main__':
    train()
