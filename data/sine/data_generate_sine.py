# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(0)

############
# Settings #
############

# dir names
folder_outer = 'data/'
folder_inner = 'sine/'
filename = 'sine_original'           # output name

# dataset size
size_normal = 10000
size_faulty = size_normal

# sine range
sine_range_x = [0.0, 2.0 * np.pi]
sine_range_y = [-1.0, 1.0]

#############
# Auxiliary #
#############


def create_sine(range_x, range_y, size):
    # Generate samples
    data = np.zeros((size, 3))
    sample_xs = np.random.uniform(range_x[0], range_x[1], size=size)
    sample_ys = np.random.uniform(range_y[0], range_y[1], size=size)

    # Store samples
    data[:, 0] = sample_xs
    data[:, 1] = sample_ys

    # Derive class
    temp = sample_ys < np.sin(sample_xs)

    temp[temp == True] = 1          # fault if below the curve
    temp[temp == False] = 0         # normal if above (or on) the curve
    data[:, 2] = temp

    # Return
    return data

########
# Main #
########


# Output files
data_filename = folder_outer + folder_inner + filename + '.csv'
pic_filename = folder_outer + folder_inner + filename + '.png'

# Dataset size
size = size_normal + size_faulty + 500

# Generate dataset
data = create_sine(sine_range_x, sine_range_y, size)

# Distinguish normal and faulty samples
data_normal = data[data[:, 2] == 0]
data_faulty = data[data[:, 2] == 1]

# Discard extra samples
data_normal = data_normal[:size_normal, :]
data_faulty = data_faulty[:size_faulty, :]

# Append and shuffle
data = np.append(data_normal, data_faulty, axis=0)
np.random.shuffle(data)

# Plot
colors = np.array(['red' if c else 'green' for c in data[:, 2]])
plt.scatter(data[:, 0], data[:, 1], c=colors)
plt.plot()
plt.xlabel('x1')
plt.ylabel('x2')
# plt.savefig(pic_filename)
plt.show()

# Store
df = pd.DataFrame(data, columns=['x1', 'x2', 'class'])
# df.to_csv(data_filename, index=False)
