import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import r2_score

class BaseModule(nn.Module):

    ####
    # Methods to be used within the Trainer interface, see trainer.py
    ####

    def training_step(self, batch, batch_idx, device):
        """ Specifies the behaviour of a single train step. """
        
        x, y = batch

        if torch.cuda.is_available():
            x = x.to(device)
            y = y.to(device)

        out = self(x)
        loss = self.loss(out, y)
        score = self.score(y, out.detach().numpy())
        return loss, score

    
    def validation_step(self, batch, batch_idx, device):
        """ Specifies the behaviour for a single validation step. """
        
        x, y = batch

        if torch.cuda.is_available():
            x = x.to(device)
            y = y.to(device)

        out = self(x)
        loss = self.loss(out, y)
        score = self.score(y, out.detach().numpy())
        return loss, score

    def predict_step(self, batch, batch_idx, device):
        """ Specifies the behaviour for a single prediction step. """
        
        x = batch

        if torch.cuda.is_available():
            x = x.to(device)

        out = self(x)
        return out

    def test_step(self, batch, batch_idx):
        """ Specifies the behaviour for a single test step. """
        
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self, optim, lr):
        """ Set the adam optimizer as optimizer. """
        if optim == 'adam':
            return torch.optim.Adam(self.parameters(), lr=lr)
        elif optim == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=lr)
        elif optim == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=lr)
        else:
            print("Optimizer not implemented.")


class Net(BaseModule):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.fc3 = nn.Linear(dim_hidden, dim_hidden)
        self.fc4 = nn.Linear(dim_hidden, dim_hidden // 4)
        self.fc5 = nn.Linear(dim_hidden // 4, dim_hidden // 16)
        self.fc6 = nn.Linear(dim_hidden // 16, 1)
        self.loss = nn.MSELoss()
        self.score = r2_score
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

class SimpleNet(BaseModule):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, 1)
        self.loss = nn.MSELoss()
        self.score = r2_score
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNetBlock(BaseModule):
    def __init__(self, nb_channels, kernel_size, batch_normalization, skip_connections):
        super().__init__()

        self.is_bn = batch_normalization
        self.is_skip = skip_connections

        self.conv1 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        self.bn1 = nn.BatchNorm2d(nb_channels)

        self.conv2 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        self.bn2 = nn.BatchNorm2d(nb_channels)

    def forward(self, x):
        y = self.conv1(x)
        if self.is_bn: y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        if self.is_bn: y = self.bn2(y)
        if self.is_skip: y = y + x
        y = F.relu(y)

        return y

