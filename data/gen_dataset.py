import csv
import pickle
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
dataset_file = os.path.join(dir_path, 'chemistry_dataset')
result_file = os.path.join(dir_path, 'results.csv')

with open(dataset_file, 'rb') as file:
    dataset = pickle.load(file)

csv_reader = csv.reader(open(result_file))

arch_code, energy = [], []
for row in csv_reader:  
    arch_code.append(row[1])
    energy.append(row[3])
arch_code.pop(0)
energy.pop(0)

for i in range(len(arch_code)):
    if arch_code[i] not in dataset:
        dataset[arch_code[i]] = eval(energy[i])

with open(dataset_file, 'wb') as file:
    pickle.dump(dataset, file)

