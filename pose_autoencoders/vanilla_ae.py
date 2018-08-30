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
from utils.log_utils import string2color, get_log_path
from utils.render_manifold import HandRenderer, plot_latent

# todo add logdir args
log_dir = "/home/dawars/projects/logdir/"

num_epochs = 5001
batch_size = 64
learning_rate = 1e-3
torch.manual_seed(7)

dataset = get_poses()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
poses = torch.Tensor(dataset).float().cuda()  # for latent space plotting

num_neurons = 20
activation = nn.ReLU


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(45, num_neurons),
            activation(True),
            nn.Linear(num_neurons, 2))
        self.decoder = nn.Sequential(
            nn.Linear(2, num_neurons),
            activation(True),
            nn.Linear(num_neurons, 3 * 15))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train():
    params = {
        'learning_rate': learning_rate,
        'batch': batch_size,
        'act': activation,
        'neurons': num_neurons,
        'weight_decay': 1e-5
    }

    model_name = "vanilla_ae"
    save_dir = get_log_path(model_name, params=params)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, save_dir))

    color = string2color(save_dir)

    to_tensor = torchvision.transforms.ToTensor()  # convert PIL image to tensor for tensorboard

    ### TRAINING ###
    model = autoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=params['weight_decay'])

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

            torch.save(model.state_dict(),
                       os.path.join(log_dir, save_dir, "{name}_{epoch}.pth".format(name=model_name, epoch=epoch)))


if __name__ == '__main__':
    train()
