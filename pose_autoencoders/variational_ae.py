import io
import os

import PIL
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import functional

from pose_autoencoders.pose_loader import get_poses
from utils.render_manifold import HandRenderer
from tensorboardX import SummaryWriter
import torchvision

#os.makedirs('./var_figures', exist_ok=True)

# todo add logdir args
log_dir = "/home/dawars/projects/logdir/variational_ae"

num_epochs = 5001
batch_size = 64
learning_rate = 1e-3
torch.manual_seed(7)

dataset = get_poses()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
poses = torch.Tensor(dataset).float().cuda()


def plot_latent(latent, bounds=(), color=(1, 0, 0)):
    plt.plot(*(latent.transpose()), '.', color=color)
    plt.xlim(*bounds)
    plt.ylim(*bounds)
    plt.axis('off')
    # plt.savefig("var_figures/pose_vae_latent_{0:04d}.png".format(epoch))
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf



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
    writer = SummaryWriter(log_dir=log_dir)
    to_tensor = torchvision.transforms.ToTensor()  # convert PIL image to tensor for tensorboard

    color = (0, 1, 0)  # todo randomize from hypterparams

    model = autoencoder().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    renderer = HandRenderer(64)

    for epoch in range(num_epochs):
        for data in dataloader:
            img = data.float()
            # img = img.view(img.size(0), -1)
            img = Variable(img).cuda()

            # ===================forward=====================
            output, mean, std = model(img)
            gen_loss, latent_loss = loss_fcn(img, output, mean, std)
            loss = 5 * gen_loss + latent_loss
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================

        # plot
        if epoch % 100 == 0:
            # todo tb save util, values as dict as {dict}
            writer.add_scalar('loss/reconstruction_loss', gen_loss.item(), epoch)
            writer.add_scalar('loss/kld_loss', latent_loss.item(), epoch)
            print(
                'epoch [{}/{}], reconstruction loss:{:.4f}, KLD: {:.4f}'.format(epoch, num_epochs, gen_loss.item(),
                                                                                latent_loss.item()))

            latent, _ = model.encoder(poses)

            bounds = (-6, 6)
            latent_plot = plot_latent(latent.cpu().detach().numpy(), bounds=bounds, color=color)
            image = PIL.Image.open(latent_plot)
            writer.add_image("latent_space", to_tensor(image), epoch)

            manifold = renderer.render_manifold(model.decoder, filename=None, bounds=bounds, color=color)
            writer.add_image("manifold", to_tensor(manifold), epoch)

            torch.save(model.state_dict(), "./variational_autoencoder_{}.pth".format(epoch))


if __name__ == '__main__':
    train()
