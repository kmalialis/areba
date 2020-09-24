# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(0)

############
# Settings #
############

# dataset size
size_normal = 110000
size_faulty = size_normal

# sine range
sine_range_x = [0.0, 10.0]
sine_range_y = [0.0, 10.0]

#############
# Auxiliary #
#############


def create_sea(range_x, range_y, size):
    # Generate samples
    data = np.zeros((size, 3))
    sample_xs = np.random.uniform(range_x[0], range_x[1], size=size)
    sample_ys = np.random.uniform(range_y[0], range_y[1], size=size)

    # Store samples
    data[:, 0] = sample_xs
    data[:, 1] = sample_ys

    # Derive class
    temp = data[:, 0] + data[:, 1]
    temp[temp <= 7.0] = 1
    temp[temp > 7.0] = 0
    data[:, 2] = temp

    # balance dataset
    data1 = data[data[:, 2] == 1]
    data1 = data1[:50000, :]

    data0 = data[data[:, 2] == 0]
    data0 = data0[:50000, :]

    data = np.concatenate((data0, data1))

    # shuffle dataset
    np.random.shuffle(data)

    # normalise dataset
    data[:, :2] = data[:, :2] / 10.0

    # Return
    return data

########
# Main #
########


# Generate dataset
data = create_sea(sine_range_x, sine_range_y, size_normal + size_faulty)

# Plot
colors = np.array(['red' if c else 'green' for c in data[:, 2]])
plt.scatter(data[:, 0], data[:, 1], c=colors)
plt.plot()
plt.xlabel('x1')
plt.ylabel('x2')
# plt.savefig('sea.png')
plt.show()

# Store
df = pd.DataFrame(data, columns=['x1', 'x2', 'class'])
# df.to_csv('sea.csv', index=False)
