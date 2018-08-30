import io
import os

import PIL
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
import torchvision

from pose_autoencoders.pose_loader import get_poses
from utils.render_manifold import HandRenderer

# os.makedirs('./figures', exist_ok=True)

# todo add logdir args
log_dir = "/home/dawars/projects/logdir/vanilla_ae"

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


def train():
    writer = SummaryWriter(log_dir=log_dir)
    to_tensor = torchvision.transforms.ToTensor()  # convert PIL image to tensor for tensorboard

    color = (1, 0, 1)  # todo randomize from hypterparams

    model = autoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    renderer = HandRenderer(64)

    for epoch in range(num_epochs):
        for data in dataloader:
            img = data.float()
            # img = img.view(img.size(0), -1)
            img = Variable(img).cuda()

            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ===================log========================
        if epoch % 100 == 0:
            # todo tb save util, values as dict as {dict}
            writer.add_scalar('loss/reconstruction_loss', loss.item(), epoch)
            print('epoch [{}/{}], reconstruction loss:{:.4f}'.format(epoch, num_epochs, loss.item()))

            bounds = (-6, 6)

            latent = model.encoder(poses)

            latent_plot = plot_latent(latent.cpu().detach().numpy(), bounds=bounds, color=color)
            image = PIL.Image.open(latent_plot)
            writer.add_image("latent_space", to_tensor(image), epoch)

            manifold = renderer.render_manifold(model.decoder, filename=None, bounds=bounds, color=color)
            writer.add_image("manifold", to_tensor(manifold), epoch)

            torch.save(model.state_dict(), os.path.join(log_dir, "vanilla_autoencoder_{}.pth".format(epoch)))


if __name__ == '__main__':
    train()
