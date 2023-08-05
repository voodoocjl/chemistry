import json
from math import log2, ceil
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from sklearn.metrics import accuracy_score


"""
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        y = self.fc(x)
        return y
"""
torch.cuda.is_available = lambda : False

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
            )
        
    def forward(self, x):
        y = self.network(x)
        y[-1] = torch.sigmoid(y[-1])
        return y

class Enco_Conv_Net(nn.Module):
    def __init__(self, n_channels, output_dim):
        super(Enco_Conv_Net, self).__init__()
        self.features_2x2 = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=2),
            nn.Sigmoid()
            )
        self.features_4x4 = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=4),
            nn.Sigmoid()
            )
        self.classifier = nn.Linear(42, output_dim)

    def forward(self, x):
        x = self.transform(x)
        x1 = self.features_2x2(x)
        x1 = torch.mean(x1, dim=1).flatten(1)
        x2 = self.features_4x4(x)
        x2 = torch.mean(x2, dim=1).flatten(1)
        x_ = torch.cat((x1, x2), 1)
        y = self.classifier(x_)
        y[-1] = torch.sigmoid(y[-1])
        return y
    
    def transform(self, x):
        len = x[0].shape[0]
        xbar = torch.cat((x[:, 6:], x[:, :6]), 1)
        x = x.unsqueeze(1).unsqueeze(1)
        xbar = xbar.unsqueeze(1).unsqueeze(1)
        x = torch.cat((x, xbar, x, xbar), 2)
        return x


class Classifier:
    def __init__(self, samples, input_dim, node_id):
        assert type(samples) == type({})
        assert input_dim     >= 1

        self.samples          = samples
        self.input_dim        = input_dim
        self.training_counter = 0
        self.node_layer       = ceil(log2(node_id + 2) - 1)
        # self.hidden_dims      = [6, 7, 8, 9, 10]  #[16, 20, 24, 28, 32]
        # self.model            = Encoder(input_dim, self.hidden_dims[self.node_layer], 2)
        self.model            = Enco_Conv_Net(4, 2)
        if torch.cuda.is_available():
            self.model.cuda()
        self.loss_fn          = nn.MSELoss()
        self.l_rate           = 0.001
        self.optimizer        = optim.Adam(self.model.parameters(), lr=self.l_rate, betas=(0.9, 0.999), eps=1e-08)
        self.epochs           = []
        self.training_accuracy = [0]
        self.boundary         = -1
        self.nets             = None
        self.maeinv           = None
        self.labels           = None
        self.random_mean      = 0

    def get_label(self, energy, fixed_mean = None):
        label = torch.zeros_like(energy)
        for i in range(energy.shape[0]):
            if fixed_mean:
                 label[i] = energy[i] > fixed_mean
            else:
                label[i] = energy[i] > energy.mean()
        return label
    
    def update_samples(self, latest_samples, root = None):
        assert type(latest_samples) == type(self.samples)
        sampled_nets = []        
        nets_maeinv  = []
        for k, v in latest_samples.items():
            net = json.loads(k)
            sampled_nets.append(net)            
            nets_maeinv.append(v)
        self.nets = torch.from_numpy(np.asarray(sampled_nets, dtype=np.float32).reshape(-1, self.input_dim))
        self.maeinv = torch.from_numpy(np.asarray(nets_maeinv, dtype=np.float32).reshape(-1, 1))
        # root has different labels
        if root:
            if self.random_mean == 0:           #first training
                self.random_mean = self.maeinv.mean()
            self.labels = self.get_label(self.maeinv, self.random_mean)
        else:
            self.labels = self.get_label(self.maeinv)
        # self.labels = self.get_label(self.maeinv)
        self.samples = latest_samples
        if torch.cuda.is_available():
            self.nets = self.nets.cuda()
            self.labels = self.labels.cuda()


    def train(self):
        if self.training_counter == 0:
            self.epochs = 20000            
        else:
            self.epochs = 3000
        self.training_counter += 1
        # in a rare case, one branch has no networks
        if len(self.nets) == 0:
            return
        for epoch in range(self.epochs):
            nets = self.nets
            labels = self.labels
            maeinv = self.maeinv
            # clear grads
            self.optimizer.zero_grad()
            # forward to get predicted values
            outputs = self.model(nets)
            # loss_s = self.loss_fn(outputs[:, :6], nets[:, 6:])
            loss_mae = self.loss_fn(outputs[:, 0], maeinv.reshape(-1))
            loss_t = self.loss_fn(outputs[:, -1], labels.reshape(-1))
            loss = loss_mae + loss_t
            loss.backward()  # back props
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()  # update the parameters

        # training accuracy 
        pred = self.model(nets).cpu()
        pred_label = (pred[:, -1] > 0.5).float()
        labels = self.labels.reshape(-1).cpu()
        acc = accuracy_score(pred_label.numpy(), labels.numpy())
        self.training_accuracy.append(acc)


    def predict(self, remaining):
        assert type(remaining) == type({})
        remaining_archs = []
        for k, v in remaining.items():
            net = json.loads(k)
            remaining_archs.append(net)
        remaining_archs = torch.from_numpy(np.asarray(remaining_archs, dtype=np.float32).reshape(-1, self.input_dim))
        if torch.cuda.is_available():
            remaining_archs = remaining_archs.cuda()

        outputs = self.model(remaining_archs)
        labels = outputs[:, -1].reshape(-1, 1)  #output labels
        xbar = outputs[:, 0].mean().detach().tolist()

        if torch.cuda.is_available():
            remaining_archs = remaining_archs.cpu()
            labels         = labels.cpu()
        result = {}
        for k in range(0, len(remaining_archs)):
            arch = remaining_archs[k].detach().numpy().astype(np.int32)
            arch_str = json.dumps(arch.tolist())
            result[arch_str] = labels[k].detach().numpy().tolist()[0]
        assert len(result) == len(remaining)
        return result, xbar
    
    def root_predict(self, remaining):
        assert type(remaining) == type({})
        remaining_archs = []
        for k, v in remaining.items():
            net = json.loads(k)
            remaining_archs.append(net)
        remaining_archs = torch.from_numpy(np.asarray(remaining_archs, dtype=np.float32).reshape(-1, self.input_dim))
        if torch.cuda.is_available():
            remaining_archs = remaining_archs.cuda()

        outputs = self.model(remaining_archs)        
        energy = outputs[:, 0]
        labels = self.get_label(energy, self.random_mean)

        if torch.cuda.is_available():
            remaining_archs = remaining_archs.cpu()
            labels         = labels.cpu()
        result = {}
        for k in range(0, len(remaining_archs)):           
            arch = remaining_archs[k].detach().numpy().astype(np.int32)
            arch_str = json.dumps(arch.tolist())
            result[arch_str] = labels[k].detach().numpy().tolist()
        assert len(result) == len(remaining)
        return result


    def split_predictions(self, remaining, method = None):
        assert type(remaining) == type({})
        samples_badness = {}
        samples_goodies = {}
        xbar = 0
        if len(remaining) == 0:
            return samples_goodies, samples_badness, 0
        if method == None:
            predictions, xbar = self.predict(remaining)  # arch_str -> pred_test_mae
            for k, v in predictions.items():
                if v < 0.5:
                    samples_badness[k] = v
                else:
                    samples_goodies[k] = v
        # elif method == 'root':
        #     predictions = self.root_predict(remaining)  # to split root node
        #     for k, v in predictions.items():
        #         if k < self.random_mean:
        #             samples_badness[k] = v
        #         else:
        #             samples_goodies[k] = v
        else:
            predictions = np.mean(list(remaining.values()))     # to split validation set
            for k, v in remaining.items():
                if v > predictions:
                    samples_badness[k] = v
                else:
                    samples_goodies[k] = v
                
        assert len(samples_badness) + len(samples_goodies) == len(remaining)
        return samples_goodies, samples_badness, xbar

    """
    def predict_mean(self):
        if len(self.nets) == 0:
            return 0
        # can we use the actual maeinv?
        outputs = self.model(self.nets)
        pred_np = None
        if torch.cuda.is_available():
            pred_np = outputs.detach().cpu().numpy()
        else:
            pred_np = outputs.detach().numpy()
        return np.mean(pred_np)
    """

    def sample_mean(self):
        if len(self.nets) == 0:
            return 0
        outputs = self.maeinv
        true_np = None
        if torch.cuda.is_available():
            true_np = outputs.cpu().numpy()
        else:
            true_np = outputs.numpy()
        return np.mean(true_np)


    def split_data(self, f1 = None):
        samples_badness = {}
        samples_goodies = {}
        if len(self.nets) == 0:
            return samples_goodies, samples_badness        
        self.train()
        outputs = self.model(self.nets)[:, -1].reshape(-1, 1)
        if torch.cuda.is_available():
            self.nets = self.nets.cpu()
            outputs   = outputs.cpu()
        predictions = {}
        for k in range(0, len(self.nets)):
            arch = self.nets[k].detach().numpy().astype(np.int32)
            arch_str = json.dumps(arch.tolist())
            predictions[arch_str] = outputs[k].detach().numpy().tolist()[0]  # arch_str -> pred_test_mae
        assert len(predictions) == len(self.nets)
        # avg_maeinv = self.sample_mean()
        # self.boundary = avg_maeinv
        for k, v in predictions.items():
            if v < 0.5:
                samples_badness[k] = self.samples[k]  # (val_loss, test_mae)
            else:
                samples_goodies[k] = self.samples[k]  # (val_loss, test_mae)
        assert len(samples_badness) + len(samples_goodies) == len(self.samples)
        return samples_goodies, samples_badness