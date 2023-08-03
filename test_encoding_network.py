import random
import pickle
import csv
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time
from sklearn.metrics import accuracy_score, f1_score

# torch.cuda.is_available = lambda : False
# torch.set_num_threads(4)

# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            # nn.Linear(64, hidden_dim),
            # nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)            
            )
        
    def forward(self, x):
        y = self.network(x)
        # y[:,-1] = torch.sigmoid(y[:,-1])
        # y = torch.round(y)
        return y


# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

with open('data/chemistry_dataset', 'rb') as file:
    dataset = pickle.load(file)

arch_code, energy = [], []
for key in dataset:
    arch_code.append(eval(key))
    energy.append(dataset[key])

# # read code, val loss and test mae from .csv file
# csv_reader = csv.reader(open('results/training.csv'))
# arch_code, energy = [], []
# for row in csv_reader:
#     arch_code.append(eval(row[1]))
#     energy.append(eval(row[3]))

def get_label(energy):
    label = torch.zeros_like(energy)
    for i in range(energy.shape[0]): 
        label[i] = energy[i] < energy.mean()
    return label

true_label = get_label(torch.tensor(energy))
t_size = 2000
arch_code_train = torch.from_numpy(np.asarray(arch_code[:t_size], dtype=np.float32))
energy_train = torch.from_numpy(np.asarray(energy[:t_size], dtype=np.float32))
label = get_label(energy_train)


if torch.cuda.is_available():
    arch_code_train = arch_code_train.cuda()
    energy_train = energy_train.cuda()
    label = label.cuda()

dataset = TensorDataset(arch_code_train, label)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

arch_code_test = torch.from_numpy(np.asarray(arch_code[t_size:], dtype=np.float32))
# energy_test = torch.from_numpy(np.asarray(energy[2000:], dtype=np.float32))
test_label = true_label[t_size:]

if torch.cuda.is_available():
    arch_code_test = arch_code_test.cuda()
    # energy_test = energy_test.cuda()
    test_label = test_label.cuda()

dataset1 = TensorDataset(arch_code_test, test_label)
dataloader1 = DataLoader(dataset1, batch_size=1000, shuffle=True)

print("dataset size: ", len(energy))
print("training size: ", len(energy_train))
print("test size: ", len(arch_code_test))

for hidden_dim in range(64, 128, 16):
    model = Encoder(12, hidden_dim, 1)
    if torch.cuda.is_available():
        model.cuda()    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_loss_list, test_loss_list = [], []
    s = time.time()
    for epoch in range(1, 2001):
        for x, y in dataloader:
            model.train()
            pred = model(x)  # shape: (2284, 21)
            # y = (y - 0.2 * torch.ones_like(y)).abs()
            # loss_s = loss_fn(pred[:, :12], x)
            # loss_s = loss_fn(pred[:, :6], x[:, 6:])
            loss_e = loss_fn(pred[:, -1], y)
            
            train_loss = loss_e #+ 0.1 * loss_s            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()        

        if epoch % 500 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(arch_code_train).cpu()
                # error = (pred[:,-1] - energy_train).abs().mean()
                # pred_label = get_label(pred[:, -1])
                pred_label = (pred[:, -1] > 0.5).float()
                label = label.cpu()
                acc = accuracy_score(pred_label.numpy(), label.numpy())
                # acc = f1_score(label, pred_label)
                print(epoch, acc)
                train_loss_list.append(acc)
    e = time.time()
    print('time: ', e-s)

    model.eval()
    with torch.no_grad():
        pred = model(arch_code_test).cpu()
        # error = (pred[:,-1] - energy_train).abs().mean()
        # error = loss_fn(pred[:,-1], energy_train)                
        # pred_label = get_label(pred[:, -1])
        pred_label = (pred[:, -1] > 0.5).float()
        test_label = test_label.cpu()
        acc = accuracy_score(pred_label.numpy(), test_label.numpy())
        # acc = f1_score(test_label, pred_label)
        print("test acc:", acc)
        train_loss_list.append(acc)
    print(train_loss_list)

plt.plot(range(len(train_loss_list)), train_loss_list, 'ro-')
plt.title('min test loss')
plt.xlabel('hidden dim')
plt.ylabel('test loss')
plt.show()
