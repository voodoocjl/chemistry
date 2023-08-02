import torch 
import time 
# Define a neural network 
class Net(torch.nn.Module): 
    def __init__(self): 
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1000, 10000)
        self.fc2 = torch.nn.Linear(10000, 10000)
    
    def forward(self, x): 
        x = self.fc1(x)
        x = self.fc2(x)
        return x

#  Create a random input tensor 
x = torch.randn(1000, 1000) 

# Create an instance of the network 
net = Net() 
# Time the forward pass on CPU 
start_time = time.time() 
for i in range(10):
    y = net(x) 
print('CPU time:', time.time() - start_time)

 # Move the network to the GPU 
net.cuda()
x = x.cuda() 
# Time the forward pass on GPU 
start_time = time.time()
for i in range(10):
    y = net(x) 
print('GPU time:', time.time() - start_time)