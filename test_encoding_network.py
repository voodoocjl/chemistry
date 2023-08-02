import random
import pickle
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

"""
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        y = self.classifier(x)
        return y
"""

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
        x1 = self.features_2x2(x)
        x1 = torch.mean(x1, dim=1).flatten(1)
        x2 = self.features_4x4(x)
        x2 = torch.mean(x2, dim=1).flatten(1)
        x_ = torch.cat((x1, x2), 1)
        y = self.classifier(x_)
        y = torch.sigmoid(y)
        return y

def get_label(energy):
    label = torch.zeros_like(energy)
    for i in range(len(energy)):
        if energy[i] < energy.mean():
            label[i] = 1
    return label


# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

with open('data/chemistry_dataset', 'rb') as file:
    dataset = pickle.load(file)
arch_code_len = len(eval(list(dataset.keys())[0]))

arch_code, energy = [], []
for k, v in dataset.items():
    arch_code_2d = []
    arch = eval(k)
    for _ in range(2):
        arch_code_2d.append(arch)
        arch_code_2d.append(arch[arch_code_len//2:]+arch[:arch_code_len//2])
    arch_code.append(arch_code_2d)
    energy.append(v)

training_size = 2000
arch_code_train = torch.from_numpy(np.asarray(arch_code[:training_size], dtype=np.float32)).unsqueeze(dim=1)  # (2000, 1, 4, 12)
energy_train = torch.from_numpy(np.asarray(energy[:training_size], dtype=np.float32))
label_train = get_label(energy_train)

train_data = TensorDataset(arch_code_train, label_train)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

arch_code_test = torch.from_numpy(np.asarray(arch_code[training_size:], dtype=np.float32)).unsqueeze(dim=1)
label_test = get_label(torch.tensor(energy, dtype=torch.float32))[training_size:]

test_acc_list = []
for n_channels in range(2, 65, 4):
    print("\nn_channels -", n_channels)
    model = Enco_Conv_Net(n_channels, 1)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_acc_list = []
    s = time.time()
    for epoch in range(1, 2001):
        model.train()
        for x, y in train_loader:
            pred = model(x)

            # loss_s = loss_fn(pred[:, :12], x)
            loss_e = loss_fn(pred[:, -1], y)

            train_loss = loss_e #+ 0.5 * loss_s
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        if epoch % 200 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(arch_code_train)
                pred_label = (pred[:, -1] > 0.5).float()
                label_train = label_train
                acc = accuracy_score(label_train, pred_label)
                f1 = f1_score(label_train, pred_label)
                print(epoch, acc, f1)
                train_acc_list.append(acc)
    e = time.time()
    print('time:', e-s)

    model.eval()
    with torch.no_grad():
        pred = model(arch_code_test)
        pred_label = (pred[:, -1] > 0.5).float()
        acc = accuracy_score(label_test, pred_label)
        f1 = f1_score(label_test, pred_label)
        print("test acc:", acc, f1)
        test_acc_list.append(acc)
        
print("max test acc:", max(test_acc_list), "n_channels:", test_acc_list.index(max(test_acc_list))+1)
plt.figure()
plt.plot(range(1, 65), test_acc_list)
plt.show()
