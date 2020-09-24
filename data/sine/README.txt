README

#####################
# sine_original.csv #
#####################
 
Consider y = sin(x) where: 
	sine_range_x = [0.0, 2.0 * np.pi]
	sine_range_y = [-1.0, 1.0]

Points below the curve are positive (y=1) and above or on the curve are negative (y=0).

The dataset contains 10K instances from each class.

Created by the code in data_generate_sine.py

############
# sine.csv #
############

The pre-processed version of the previous file => use this one.

Pre-processing steps are:
- feature scaling to [0,1]
- random shuffle of the dataset

Created by the code in data_preprocess_sine.py
