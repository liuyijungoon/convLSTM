from MovingMNIST import MovingMNIST
import torch
import pdb
train_set = MovingMNIST(root='./data/moving_mnist', train=True, download=False)
test_set = MovingMNIST(root='./data/moving_mnist_test', train=False, download=False)
# pdb.set_trace()
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=100,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                 dataset=test_set,
                 batch_size=100,
                 shuffle=True)