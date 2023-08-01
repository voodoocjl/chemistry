import pickle
import random

with open('search_space', 'rb') as file:
    search_space = pickle.load(file)
# random.shuffle(search_space)
print(len(search_space))

# training_data = search_space[:10000]

# with open('train_space_1', 'wb') as file:
#     pickle.dump(training_data, file)

training_data = search_space[80000:]

with open('train_space_3', 'wb') as file:
    pickle.dump(training_data, file)

# with open('search_space_shuffle', 'wb') as file:
#     pickle.dump(search_space, file)