# -*- coding: utf-8 -*-

import numpy as np
from class_qbr import QBR
from class_areba import AREBA
from class_sota import Baseline, CS, Sliding, OOB

###########################################################################################
#                                     Auxiliary functions                                 #
###########################################################################################


# Update prequential evaluation metric (recall / specificity) using fading factor
def update_preq_metric(s_prev, n_prev, correct, fading_factor):
    s = correct + fading_factor * s_prev
    n = 1.0 + fading_factor * n_prev
    metric = s / n

    return s, n, metric


# Update delayed evaluation metric (size, recall / specificity)
def update_delayed_metric(prev, flag, forget_rate):
    return (1.0 - forget_rate) * flag + forget_rate * prev


# Get next example
def draw_example(random_state, df):
    idx = random_state.choice(df.index, size=1)[0]
    next = df.iloc[idx, :].values
    df_new = df.drop(idx, axis=0).reset_index(drop=True)

    return df_new, next


# Get next example on condition
def draw_example_conditional(random_state, df, val_x, prob_x):
    r = random_state.rand()
    if r <= prob_x:
        df_temp = df[df['x1'] < val_x]
    else:
        df_temp = df[df['x1'] >= val_x]

    idx = random_state.choice(df_temp.index, size=1)[0]
    next = df.iloc[idx, :].values
    df_new = df.drop(idx, axis=0).reset_index(drop=True)

    return df_new, next


# Find decay step for gradual drift
def find_decay(pre_val, post_val, drift_time_start, drift_time_stop):
    return (post_val - pre_val) / (drift_time_stop - drift_time_start + 1.0)


# revert class
def revert_df_class(df, target):
    # revert labels
    srs = df[target]
    srs.replace(to_replace=1.0, value=-1.0, inplace=True)
    srs.replace(to_replace=0.0, value=1.0, inplace=True)
    srs.replace(to_replace=-1.0, value=0.0, inplace=True)

    # create new df
    df_new = df.drop(target, axis=1)
    df_new[target] = srs

    return df_new


# revert ground truth
def revert_truth(arr):
    # get ground truth
    truth = arr[0][0]   # arr is of shape (1,1)

    # revert
    if truth == 0.0:    # if negative
        new_y = 1.0
        example_neg = 0
    else:               # if positive
        new_y = 0.0
        example_neg = 1

    arr_new = np.array([new_y]).reshape(1, 1)

    return arr_new, example_neg

###########################################################################################
#                                           Run                                           #
###########################################################################################


def run(random_state, time_steps, df_neg, df_pos, models, method, prob_pos, preq_fading_factor, layer_dims,
        cs_update_freq, cs_upper_weight, sliding_window_size, queue_size_budget, delayed_forget_rate,
        flag_drift, drift_type, drift_speed, time_drift_start, time_drift_stop_gradual, post_prob_pos,
        val_x, prob_x, post_prob_x, target, flag_noisy_truth, prob_noisy_truth):

    ############################
    # Init prequential metrics #
    ############################

    preq_recalls = np.zeros(time_steps)
    preq_specificities = np.zeros(time_steps)
    preq_gmeans = np.zeros(time_steps)

    preq_recall, preq_specificity = (1.0,) * 2  # NOTE: init to 1.0 not 0.0
    preq_recall_s, preq_recall_n = (0.0,) * 2
    preq_specificity_s, preq_specificity_n = (0.0,) * 2

    ########################
    # Init delayed metrics #
    ########################

    # size
    delayed_size_neg, delayed_size_pos = (0.0,) * 2

    ################
    # Init methods #
    ################

    technique = Baseline(models[0])     # Baseline init

    # State-of-the-art
    if method == 'cs' or method == 'adaptive_cs':
        technique = CS(models[0], cs_update_freq, cs_upper_weight)
    elif method == 'sliding':
        technique = Sliding(models[0], sliding_window_size)
    elif method == 'oob_single' or method == 'oob':
        technique = OOB(models)
    # Proposed
    elif method == 'qbr':
        technique = QBR(models[0], queue_size_budget)
    elif method == 'areba':
        technique = AREBA(models[0], queue_size_budget)

    ######################
    # Init concept drift #
    ######################

    drift_gradual_decay = 0.0
    if flag_drift and drift_speed == 'gradual':
        if drift_type == 'prior':
            drift_gradual_decay = find_decay(prob_pos, post_prob_pos, time_drift_start, time_drift_stop_gradual)
        elif drift_type == 'likelihood':
            drift_gradual_decay = find_decay(prob_x, post_prob_x, time_drift_start, time_drift_stop_gradual)

    #########
    # Start #
    #########

    for t in range(time_steps):
        if t % 1000 == 0:
            print('Time step: ', t)

        #################
        # Concept drift #
        #################

        if flag_drift:
            if not (drift_type == 'posterior'):
                if t == time_drift_start and drift_speed == 'abrupt':
                    if drift_type == 'prior':
                        prob_pos = post_prob_pos
                    elif drift_type == 'likelihood':
                        prob_x = post_prob_x

                elif time_drift_start <= t <= time_drift_stop_gradual and drift_speed == 'gradual':
                    if drift_type == 'prior':
                        prob_pos += drift_gradual_decay
                    elif drift_type == 'likelihood':
                        prob_x += drift_gradual_decay
            else:
                if t == time_drift_start and drift_speed == 'abrupt':
                    df_pos_temp = revert_df_class(df_neg, target)
                    df_neg = revert_df_class(df_pos, target)
                    df_pos = df_pos_temp

            # reset preq. (not delayed) metrics for plotting purposes
            if (t == time_drift_start and drift_speed == 'abrupt') or \
                    ((t == time_drift_start or t == time_drift_stop_gradual) and drift_speed == 'gradual'):
                preq_recall, preq_recall_s, preq_recall_n = (0.0,) * 3
                preq_specificity, preq_specificity_s, preq_specificity_n = (0.0,) * 3

        ####################
        # Get next example #
        ####################

        if random_state.rand() > prob_pos:
            if (not flag_drift) or (flag_drift and (drift_type == 'prior' or drift_type == 'posterior')):
                df_neg, next = draw_example(random_state, df_neg)

            if flag_drift and drift_type == 'likelihood':
                df_neg, next = draw_example_conditional(random_state, df_neg, val_x, prob_x)

            example_neg = True
        else:
            # positive example
            df_pos, next = draw_example(random_state, df_pos)
            example_neg = False

        ##############
        # Prediction #
        ##############

        # get x
        x = next[:-1]
        x = x.reshape(1, x.shape[0])

        # predict
        _, y_hat_class = technique.predict(x)

        ####################
        # Get ground truth #
        ####################

        # get ground truth
        y = next[-1]
        y = np.array(y).reshape(1, 1)

        #######################
        # Update preq metrics #
        #######################

        # check if misclassification
        correct = 0
        if y == y_hat_class:
            correct = 1

        # update preq. recall / specificity
        if example_neg:
            preq_specificity_s, preq_specificity_n, preq_specificity = update_preq_metric(preq_specificity_s,
                                                                                          preq_specificity_n, correct,
                                                                                          preq_fading_factor)
        else:
            preq_recall_s, preq_recall_n, preq_recall = update_preq_metric(preq_recall_s, preq_recall_n, correct,
                                                                           preq_fading_factor)

        preq_gmean = np.sqrt(preq_recall * preq_specificity)

        #  append to results
        preq_recalls[t] = preq_recall
        preq_specificities[t] = preq_specificity
        preq_gmeans[t] = preq_gmean

        ######################
        # Noisy ground truth #
        ######################
        # this piece of code must be placed here as the prequential metrics must remain unaffected

        if flag_noisy_truth:
            if random_state.rand() <= prob_noisy_truth:
                y, example_neg = revert_truth(y)

        ##########################
        # Update delayed metrics #
        ##########################

        delayed_size_neg = update_delayed_metric(delayed_size_neg, example_neg, delayed_forget_rate)
        delayed_size_pos = update_delayed_metric(delayed_size_pos, not example_neg, delayed_forget_rate)

        ####################
        # State-of-the-art #
        ####################

        # Sliding window
        if method == 'sliding':
            x, y = technique.append_to_win(x, y, layer_dims[0])

        # Adaptive cs
        if method == 'adaptive_cs':
            if (delayed_size_pos < delayed_size_neg) and (delayed_size_pos != 0.0):
                imbalance_rate = delayed_size_neg / delayed_size_pos
                technique.adapt_costs(t, new_cs_weight_neg=1.0, new_cs_weight_pos=imbalance_rate)
            elif (delayed_size_neg < delayed_size_pos) and (delayed_size_neg != 0.0):
                imbalance_rate = delayed_size_pos / delayed_size_neg
                technique.adapt_costs(t, new_cs_weight_neg=imbalance_rate, new_cs_weight_pos=1.0)

        # OOB
        if method == 'oob_single' or method == 'oob':
            # Calculate class imbalance rate
            imbalance_rate = 1.0
            if (not example_neg) and (delayed_size_pos < delayed_size_neg) and (delayed_size_pos != 0.0):
                imbalance_rate = delayed_size_neg / delayed_size_pos
            elif example_neg and (delayed_size_neg < delayed_size_pos) and (delayed_size_neg != 0.0):
                imbalance_rate = delayed_size_pos / delayed_size_neg

            # OOB oversample
            technique.oob_oversample(random_state, imbalance_rate)

        ###############
        # QBR / AREBA #
        ###############

        if method == 'qbr' or method == 'areba':
            technique.append_to_queues(x, y)

            if method == 'areba':
                technique.adapt_queues(delayed_size_neg, delayed_size_pos)

            x, y = technique.get_training_set(layer_dims[0])

        ####################
        # Train classifier #
        ####################

        technique.train(x, y)

    return preq_recalls, preq_specificities, preq_gmeans
