# -*- coding: utf-8 -*-

####################################################################################################
# README                                                                                           #
# This is the code for the Adaptive REBAlancing (AREBA) algorithm.                                 #
#                                                                                                  #
# PAPER PDF                                                                                        #
#   https://ieeexplore.ieee.org/document/9203853 or                                                #
#   https://doi.org/10.1109/TNNLS.2020.3017863                                                     #
#                                                                                                  #
# CITATION REQUEST                                                                                 #
# If you have found part of our work useful please cite it as follows:                             #
#                                                                                                  #
#   K. Malialis, C. G. Panayiotou and M. M. Polycarpou,                                            #
#   "Online Learning With Adaptive Rebalancing in Nonstationary Environments,"                     #
#   in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2020.3017863. #
#                                                                                                  #
# INSTRUCTIONS                                                                                     #
# 1. Set the values for the parameters found under the following sections:                         #
#   "Settings A: Scenario",                                                                        #
#   "Settings B: Concept drift" and                                                                #
#   "Settings C: Methods"                                                                          #
# 2. Run this file and the results will be generated in the "exps" folder. For example, if you run #
#    this file as is, it will generate the results for AREBA_20 shown in Fig. 9 of the paper.      #
# 3. Use the "plot_results.py" to plot the results.                                                #
#                                                                                                  #
# SOFTWARE                                                                                         #
# The code has been generated and tested with the following:                                       #
#   Python 3.7                                                                                     #
#   tensorflow 1.13.2                                                                              #
#   Keras 2.2.4                                                                                    #
#   numpy 1.17.4                                                                                   #
#   pandas 0.25.3                                                                                  #
####################################################################################################

import numpy as np
import pandas as pd
from run_main_synthetic import run
from class_nn_standard import NN_standard

###########################################################################################
#                                     Auxiliary functions                                 #
###########################################################################################


# Create text file
def create_file(filename):
    f = open(filename, 'w')
    f.close()


# Write array to a row in the given file
def write_to_file(filename, arr):
    with open(filename, 'a') as f:
        np.savetxt(f, [arr], delimiter=', ', fmt='%1.6f')

###########################################################################################
#                                           Main                                          #
###########################################################################################


def main():

    # Reproducibility
    seed = 0
    random_state = np.random.RandomState(seed)

    ########################
    # Settings A: Scenario #
    ########################

    # time steps & repetitions
    repeats = 10  # number of repetitions
    times = 5000  # time steps per repetition

    # Dataset
    dataset = 'sine'  # 'sine', 'circle', 'sea'
    target = 'class'

    # class imbalanace method
    method = 'areba'  # 'baseline', 'cs', 'sliding', 'adaptive_cs', 'oob_single', 'oob', 'qbr', 'areba'

    # class imbalance rate
    prob_pos = 0.01

    # noisy ground truth
    flag_noisy_truth = 0
    prob_noisy_truth = 0.1

    # fixed - do not alter the following for reproducibility

    # Prequential evaluation
    preq_fading_factor = 0.99  # 0 << f < 1.0 - typically, >= 0.8

    # Delayed size metric
    delayed_forget_rate = preq_fading_factor

    # store results
    flag_store = 1

    #############################
    # Settings B: Concept drift #
    #############################

    flag_drift = False

    time_drift_start = 10000  # time_drift_stop_abrupt = time_drift_start
    time_drift_stop_gradual = 3000

    drift_type = 'likelihood'  # 'prior', 'likelihood', 'posterior'
    drift_speed = 'abrupt'  # 'abrupt', 'gradual'

    # fixed - do not alter the following for reproducibility

    if flag_drift and drift_type == 'posterior' and drift_speed == 'gradual':
        raise Exception('Drift setting not currently implemented.')

    # prior drift
    post_prob_pos = 1.0 - prob_pos

    # likelihood drift
    if dataset == 'sine':
        val_x = 0.6
    elif dataset == 'circle':
        val_x = 0.4
    elif dataset == "sea":
        val_x = 0.5

    prob_x = 0.9  # p(x < val_x | neg)
    post_prob_x = 1.0 - prob_x

    #######################
    # Settigns C: Methods #
    #######################

    # AREBA or QBR: budget >= 2
    queue_size_budget = 20

    # fixed - do not alter the following for reproducibility

    # Baseline
    learning_rate = 0.01
    output_activation = 'sigmoid'
    loss_function = 'binary_crossentropy'
    weight_init = "he"
    class_weights = {0: 1.0, 1: 1.0}
    num_epochs = 1
    minibatch_size = 1
    layer_dims = [2, 8, 1]

    # Adaptive CS: for stability
    cs_update_freq = 250
    cs_upper_weight = 50

    # Sliding: window size
    sliding_window_size = 100

    # OOB: number of classifiers
    ensemble_size = 20

    # safety check
    if method == 'oob_single':
        ensemble_size = 1

    ################
    # Output files #
    ################

    # output directory
    out_dir = 'exps/' + dataset + '/prob_pos_' + str(int(prob_pos * 100)) + '/'
    if flag_drift:
        out_dir = 'exps/' + dataset + '/prob_pos_' + str(int(prob_pos * 100)) + '_drift_' + \
                  str(drift_type) + '_' + str(drift_speed) + '/'
    out_dir = 'exps/'

    # output filenames
    out_name = method
    if method == 'qbr' or method == 'areba':
        out_name += str(queue_size_budget)

    filename_recalls = out_name + '_preq_recalls' + '.txt'
    filename_specificities = out_name + '_preq_specificities' + '.txt'
    filename_gmeans = out_name + '_preq_gmeans' + '.txt'

    # Create output files
    if flag_store:
        create_file(out_dir + filename_recalls)
        create_file(out_dir + filename_specificities)
        create_file(out_dir + filename_gmeans)

    ##############
    # Input data #
    ##############

    # Dataset dirs
    dataset_dir = ''    # init
    if dataset == 'sine':
        dataset_dir = 'data/sine/sine.csv'
    elif dataset == 'circle':
        dataset_dir = 'data/circle/circle.csv'
    elif dataset == 'sea':
        dataset_dir = 'data/sea/sea.csv'

    # Load data
    df = pd.read_csv(dataset_dir)  # must already be pre-processed

    df_neg = df[df[target] == 0]
    df_neg.reset_index(drop=True, inplace=True)

    df_pos = df[df[target] == 1]
    df_pos.reset_index(drop=True, inplace=True)

    #########
    # Start #
    #########

    for r in range(repeats):
        print('Repetition: ', r)

        # NN
        nn_standard = NN_standard(
            layer_dims=layer_dims,
            learning_rate=learning_rate,
            output_activation=output_activation,
            loss_function=loss_function,
            num_epochs=num_epochs,
            weight_init=weight_init,
            class_weights=class_weights,
            minibatch_size=minibatch_size)

        # nn_standard.model.summary()
        # for layer in nn_standard.model.layers:
        #     print(layer.get_output_at(0).get_shape().as_list())

        # model(s)
        models = [nn_standard]
        if method == 'oob':
            for i in range(ensemble_size - 1):
                models.append(
                    NN_standard(
                        layer_dims=layer_dims,
                        learning_rate=learning_rate,
                        output_activation=output_activation,
                        loss_function=loss_function,
                        num_epochs=num_epochs,
                        weight_init=weight_init,
                        class_weights=class_weights,
                        minibatch_size=minibatch_size,
                        seed=seed + i + 1))

        # start
        recall, specificity, gmean = run(random_state, times, df_neg, df_pos, models, method, prob_pos,
                                         preq_fading_factor, layer_dims, cs_update_freq, cs_upper_weight,
                                         sliding_window_size, queue_size_budget, delayed_forget_rate,
                                         flag_drift, drift_type, drift_speed, time_drift_start,
                                         time_drift_stop_gradual, post_prob_pos, val_x, prob_x, post_prob_x, target,
                                         flag_noisy_truth, prob_noisy_truth)

        # store
        if flag_store:
            write_to_file(out_dir + filename_recalls, recall)
            write_to_file(out_dir + filename_specificities, specificity)
            write_to_file(out_dir + filename_gmeans, gmean)


if __name__ == "__main__":
    main()
