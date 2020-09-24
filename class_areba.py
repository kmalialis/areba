# -*- coding: utf-8 -*-

import numpy as np
from collections import deque
from class_sota import Baseline

###########################################################################################
#                                Adaptive REBAlancing (AREBA)                             #
###########################################################################################


class AREBA(Baseline):

    ###############
    # Constructor #
    ###############

    def __init__(self, model, queue_size_budget):
        Baseline.__init__(self, model)

        # budget
        self.budget = queue_size_budget

        # init queues
        self.xs_neg = deque(maxlen=1)
        self.ys_neg = deque(maxlen=1)

        self.xs_pos = deque(maxlen=1)
        self.ys_pos = deque(maxlen=1)

    #############
    # Auxiliary #
    #############

    def adapt_queue(self, q, q_cap):
        if q == 'neg':
            self.xs_neg = deque(self.xs_neg, q_cap)
            self.ys_neg = deque(self.ys_neg, q_cap)
        elif q == 'pos':
            self.xs_pos = deque(self.xs_pos, q_cap)
            self.ys_pos = deque(self.ys_pos, q_cap)

    #######
    # API #
    #######

    def get_training_set(self, n_features):
        # merge queues
        xs = list(self.xs_neg) + list(self.xs_pos)
        ys = list(self.ys_neg) + list(self.ys_pos)

        # convert merged queues to np arrays
        size = len(ys)  # current queue size
        x = np.array(xs).reshape(size, n_features)
        y = np.array(ys).reshape(size, 1)

        #  batch GD
        self.model.change_minibatch_size(size)

        # return
        return x, y

    def append_to_queues(self, x, y):
        if y == 0:
            # append
            self.xs_neg.append(x)
            self.ys_neg.append(y)
        else:
            # append
            self.xs_pos.append(x)
            self.ys_pos.append(y)

    # AREBA mechanism
    def adapt_queues(self, delayed_size_neg, delayed_size_pos):
        length_q_pos = len(self.ys_pos)
        capacity_q_pos = self.ys_pos.maxlen

        length_q_neg = len(self.ys_neg)
        capacity_q_neg = self.ys_neg.maxlen

        if length_q_pos == 0 and capacity_q_neg < self.budget:
            self.adapt_queue('neg', capacity_q_neg + 1)
        elif length_q_neg == 0 and capacity_q_pos < self.budget:
            self.adapt_queue('pos', capacity_q_pos + 1)
        else:
            if delayed_size_neg > delayed_size_pos:
                if capacity_q_pos == length_q_pos:
                    if capacity_q_pos < self.budget / 2.0:
                        self.adapt_queue('pos', capacity_q_pos + 1)
                        self.adapt_queue('neg', capacity_q_pos)
                    elif capacity_q_pos == self.budget / 2.0 and capacity_q_neg != capacity_q_pos:
                        self.adapt_queue('neg', capacity_q_pos)

            if delayed_size_neg <= delayed_size_pos:
                if capacity_q_neg == length_q_neg:
                    if capacity_q_neg < self.budget / 2.0:
                        self.adapt_queue('neg', capacity_q_neg + 1)
                        self.adapt_queue('pos', capacity_q_neg)
                    elif capacity_q_neg == self.budget / 2.0 and capacity_q_pos != capacity_q_neg:
                        self.adapt_queue('pos', capacity_q_neg)
