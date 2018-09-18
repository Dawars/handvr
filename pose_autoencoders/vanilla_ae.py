import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from pose_autoencoders.pose_loader import get_poses
from utils.mano_utils import get_mano_vertices
from utils.render_manifold import HandRenderer

os.makedirs('./figures', exist_ok=True)

num_epochs = 1501
batch_size = 64
learning_rate = 1e-5
torch.manual_seed(7)

dataset = get_poses()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
poses = torch.Tensor(dataset).float().cuda()


def plot_latent(latent, epoch):
    plt.plot(*(latent.transpose()), '.', color='blue')
    plt.title("Vanilla Autoencoder latent space at epoch {}".format(epoch))
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.savefig("figures/pose_ae_latent_{0:04d}.png".format(epoch))
    plt.close()


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(45, 20),
            nn.ReLU(True),
            nn.Linear(20, 2))
        self.decoder = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(True),
            nn.Linear(20, 3 * 15))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def vertex_mse_loss(y, y_hat):
    shape = np.zeros([y.shape[0], 10])
    rot = torch.tensor([[0, 0, 0]], dtype=torch.float)  # global rotation
    cams = rot.view(1, 3).repeat(y.shape[0], 1).view(-1, 3).cuda()

    verts_orig = get_mano_vertices(shape, torch.cat([cams, y], dim=1), device=torch.device('cuda'))
    vertes_pred = get_mano_vertices(shape, torch.cat([cams, y_hat], dim=1), device=torch.device('cuda'))
    return nn.MSELoss()(vertes_pred, verts_orig)


def train():
    model = autoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    renderer = HandRenderer()


    for epoch in range(num_epochs):
        for data in dataloader:
            y = torch.tensor(data, dtype=torch.float).cuda()

            # ===================forward=====================
            y_hat = model(y).cuda()

            loss = vertex_mse_loss(y, y_hat)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================

        # plot
        if epoch % 10 == 0:
            print('epoch [{}/{}], reconstruction loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        if epoch % 50 == 0:
            with torch.no_grad():
                latent = model.encoder(poses)
                plot_latent(latent.cpu().numpy(), epoch)

                filename = "manifolds/vanilla/vanilla_manifold_{:04d}.png".format(epoch)
                renderer.render_manifold(model.decoder, filename)

    torch.save(model.state_dict(), './sim_autoencoder.pth')


if __name__ == '__main__':
    train()
