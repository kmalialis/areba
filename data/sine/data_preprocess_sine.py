# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#########################################################
# DESCRIPTION                                           #
#                                                       #
# The original dataset must of type dataframe.          #
#                                                       #
# Two pre-processing steps for sine:                    #
# - Feature scaling: f <- (f - f_min) / (f_max - f_min) #
# - Randomly shuffle the dataset                        #
#########################################################

# file directories
dataset_dir = 'data/sine/sine_original.csv'
output_dir = 'data/sine/sine.csv'


# # load data
# df = pd.read_csv(dataset_dir)
#
# # feature scaling
# x1_vals = (0.0, 2 * np.pi)  # (min, max) for x1
# x2_vals = (-1.0, 1.0)       # (min, max) for x2
#
# x1_min = x1_vals[0]
# x1_range = x1_vals[1] - x1_vals[0]
#
# x2_min = x2_vals[0]
# x2_range = x2_vals[1] - x2_vals[0]
#
# df.iloc[:,0] = (df.iloc[:,0] - x1_min) / float(x1_range)
# df.iloc[:,1] = (df.iloc[:,1] - x2_min) / float(x2_range)
#
# # shuffle
# df = df.sample(frac=1).reset_index(drop=True)


# NOTE
# Because the pre-processed version was generated without fixing the random seed, in order to
# visualise it I commented-out the previous and re-load the pre-processed version.

# Plot
pic_filename = 'data/sine/sine.png'
df = pd.read_csv(output_dir)
print(df.iloc[:5,:])
data = df.values

colors = np.array(['red' if c else 'green' for c in data[:,2]])
plt.scatter(data[:,0], data[:,1], c=colors)
plt.plot()
plt.xlabel('x1')
plt.ylabel('x2')
# plt.savefig(pic_filename)
plt.show()


# store
# df.to_csv(output_dir, index=False)