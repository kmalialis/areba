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
folder_inner = 'circle/'
filename = 'circle_original'           # output name

# dataset size
size_normal = 10000
size_faulty = size_normal * 7

# circle
# <(0.2, 0.5), 0.15>
# <(0.4, 0.5), 0.2>
# <(0.6, 0.5), 0.25>
# <(0.8, 0.5), 0.3
circle_originX = 0.4
circle_originY = 0.5
circle_radius = 0.2
circle_range_x = [0.0, 1.0]
circle_range_y = [0.0, 1.0]

#############
# Auxiliary #
#############


def create_circle(range_x, range_y, size):
    # Generate samples
    data = np.zeros((size, 3))
    sample_xs = np.random.uniform(range_x[0], range_x[1], size=size)
    sample_ys = np.random.uniform(range_y[0], range_y[1], size=size)

    # Store samples
    data[:, 0] = sample_xs
    data[:, 1] = sample_ys

    # Derive class
    temp = circle_radius >= np.sqrt(
        np.power(sample_xs - circle_originX, 2) +
        np.power(sample_ys - circle_originY, 2)
    )

    temp[temp == True] = 1          # fault if within circle
    temp[temp == False] = 0         # normal if outside the circle
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
size = size_normal + size_faulty + 5000

# Generate dataset
data = create_circle(circle_range_x, circle_range_y, size)

# Distinguish normal and faulty samples
data_normal = data[data[:, 2] == 0]
data_faulty = data[data[:, 2] == 1]

# Discard extra samples
data_normal = data_normal[:size_normal, :]
data_faulty = data_faulty[:size_normal, :]

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
