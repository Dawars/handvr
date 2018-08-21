import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from pose_autoencoders.pose_loader import get_poses

if not os.path.exists('./figures'):
    os.mkdir('./figures')

num_epochs = 301
batch_size = 64
learning_rate = 1e-3
torch.manual_seed(7)

dataset = get_poses('../../mpi/data/mano/MANO_RIGHT_py3.pkl')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
poses = torch.Tensor(dataset).float().cuda()


def plot_latent(latent, epoch):
    plt.plot(*(latent.transpose()), '.', color='blue')
    plt.title("Vanilla Autoencoder latent space at epoch {}".format(epoch))
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.savefig("figures/pose_ae_latent_{0:03d}.png".format(epoch))
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


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:

        img = data.float()
        # img = img.view(img.size(0), -1)
        img = Variable(img).cuda()

        # plot
        if epoch % 10 == 0:
            latent = model.encoder(poses)
            plot_latent(latent.cpu().data.numpy(), epoch)

        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], reconstruction loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.item()))

torch.save(model.state_dict(), './sim_autoencoder.pth')
