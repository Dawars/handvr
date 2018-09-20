import torch
from torch.nn.functional import _pointwise_loss

from smpl.batch_lbs import batch_rodrigues, euler2quat


class QuaternionLoss(torch.autograd.function):
    def __init__(self, ):
        super(QuaternionLoss, self).__init__()

    def forward(self, input, target):
        return torch.nn.functional.mse_loss(1-torch.dot(euler2quat(input), euler2quat(target)))


class MatrixLoss(torch.autograd.function):
    def __init__(self, ):
        super(MatrixLoss, self).__init__()

    def forward(self, input, target):
        return torch.nn.functional.mse_loss(batch_rodrigues(input), batch_rodrigues(target))


class IdentityDeviationLoss(torch.autograd.function):
    def __init__(self, ):
        super(IdentityDeviationLoss, self).__init__()

    def forward(self, input, target):
        """
        :param input: [n, n,n]
        :param target:
        :return:
        """
        loss = lambda a, b: torch.norm(torch.norm(torch.eye(3) - torch.bmm(a, b.transpose(3, 4))))
        return _pointwise_loss(loss, None, input, target)
