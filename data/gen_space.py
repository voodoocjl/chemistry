import pickle
import random

with open('search_space_3', 'rb') as file:
    search_space = pickle.load(file)
# random.shuffle(search_space)
print(len(search_space))

with open('search_space_1', 'rb') as file:
    search_space_1 = pickle.load(file)
# random.shuffle(search_space)
print(len(search_space_1))

search_space = search_space.append(search_space_1)

with open('search_space', 'wb') as file:
    pickle.dump(search_space, file)

# with open('search_space_shuffle', 'wb') as file:
#     pickle.dump(search_space, file)