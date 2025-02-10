import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim

import pandas as pd
import numpy as np
import copy

torch.manual_seed(1)


class Net(nn.Module):
    def __init__(self, inp, out, activation, num_hidden_units=100, num_layers=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inp, num_hidden_units, bias=True)
        self.fc2 = nn.ModuleList()
        for i in range(num_layers):
            self.fc2.append(nn.Linear(num_hidden_units, num_hidden_units, bias=True))
        self.fc3 = nn.Linear(num_hidden_units, out, bias=True)
        self.activation = activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        for fc in self.fc2:
            x = fc(x)
            x = self.activation(x)
        x = self.fc3(x)
        return x

    def predict(self, x):
        self.eval()
        y = self(x)
        x = x.cpu().numpy().flatten()
        y = y.cpu().detach().numpy().flatten()
        return [x, y]


def init_weights(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find("Linear") != -1:
        # apply a uniform distribution to the weights and a bias=0
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def train(lam1, loader, gradient_dim, EPOCH, BATCH_SIZE):
    state = copy.deepcopy(net.state_dict())
    best_loss = np.inf

    lossTotal = np.zeros((EPOCH, 1))
    lossRegular = np.zeros((EPOCH, 1))
    lossDerivatives = np.zeros((EPOCH, 1))
    # start training
    for epoch in range(EPOCH):
        scheduler.step()
        epoch_mse0 = 0.0
        epoch_mse1 = 0.0

        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step

            b_x = Variable(batch_x)
            b_y = Variable(batch_y)

            net.eval()
            b_x.requires_grad = True

            output0 = net(b_x)
            output0.sum().backward(retain_graph=True, create_graph=True)
            output1 = b_x.grad
            b_x.requires_grad = False

            net.train()

            mse0 = loss_func(output0, b_y[:, 0:1])
            mse1 = loss_func(output1, b_y[:, 1 : (1 + gradient_dim)])
            epoch_mse0 += mse0.item() * BATCH_SIZE
            epoch_mse1 += mse1.item() * BATCH_SIZE

            loss = mse0 + lam1 * mse1

            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

        epoch_mse0 /= num_data
        epoch_mse1 /= num_data
        epoch_loss = epoch_mse0 + lam1 * epoch_mse1

        lossTotal[epoch] = epoch_loss
        lossRegular[epoch] = epoch_mse0
        lossDerivatives[epoch] = epoch_mse1
        if epoch % 50 == 0:
            print(
                "epoch",
                epoch,
                "lr",
                "{:.7f}".format(optimizer.param_groups[0]["lr"]),
                "mse0",
                "{:.5f}".format(epoch_mse0),
                "mse1",
                "{:.5f}".format(epoch_mse1),
                "loss",
                "{:.5f}".format(epoch_loss),
            )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            state = copy.deepcopy(net.state_dict())
    # state = copy.deepcopy(net.state_dict())
    print("Best score:", best_loss)
    return state, lossTotal, lossRegular, lossDerivatives


def getDerivatives(x):
    x1 = x.requires_grad_(True)
    output = net.eval()(x1)
    nn = output.shape[0]
    gradx = np.zeros((nn, 2))
    for ii in range(output.shape[0]):
        y_def = output[ii].backward(retain_graph=True)  # noqa: F841
        gradx[ii, :] = x1.grad[ii]
    return gradx


def plotLoss(lossTotal, lossRegular, lossDerivatives):

    fig, ax = plt.subplots(1, 1, dpi=120)
    plt.semilogy(lossTotal / lossTotal[0], label="Total loss")
    plt.semilogy(lossRegular[:, 0] / lossRegular[0], label="Regular loss")
    plt.semilogy(lossDerivatives[:, 0] / lossDerivatives[0], label="Derivatives loss")
    ax.set_xlabel("epochs")
    ax.set_ylabel("L/L0")
    ax.legend()
    fig.subplots_adjust(
        left=0.1, right=0.9, bottom=0.15, top=0.9, wspace=0.3, hspace=0.2
    )
    plt.savefig("figures/Loss.png")
    plt.show()


if __name__ == "__main__":
    data_df = pd.read_csv("heat_inversion_uniform.csv")
    train_x = torch.tensor(data_df[["k1", "k2", "k3"]].values).float()
    train_y = torch.tensor(data_df[["y", "dy_dk1", "dy_dk2", "dy_dk3"]].values).float()

    x, y = Variable(train_x), Variable(train_y)

    net = Net(inp=3, out=1, activation=nn.Tanh(), num_hidden_units=256, num_layers=2)

    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    loss_func = torch.nn.MSELoss()

    BATCH_SIZE = 100
    EPOCH = 10000
    num_data = train_x.shape[0]
    torch_dataset = Data.TensorDataset(x, y)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
    )

    lam1 = 0.5
    state, lossTotal, lossRegular, lossDerivatives = train(
        lam1, loader, 3, EPOCH, BATCH_SIZE
    )
    net.load_state_dict(state)

    plotLoss(lossTotal, lossRegular, lossDerivatives)
