import numpy as np
import tensorflow as tf
import math
import copy
#import torch
#import random
#import os

class ListChecker(object):
    def __init__(self):
        self.shape_list = []
    def depth(self, inputs):
        in_shape = np.shape(inputs)
        in_len = len(inputs)
        shape_list = []
        uniq_shape = []
        for l in range(in_len):
            my_shape = np.shape(inputs[l])
            shape_list.append(my_shape)
            if my_shape not in uniq_shape:
                uniq_shape.append(my_shape)
        if len(uniq_shape)==1:
            return [in_shape]
        for l in range(in_len):
            if len(shape_list[l])>0:
                my_shape = self.depth(shape_list[l])
                shape_list[l] = my_shape
        return shape_list
    def print(self, inputs, log_tag='', w_names=[]):
        in_len = len(inputs)
        shape_list = self.depth(inputs)
        if in_len!=len(w_names):
            if len(w_names)>0:
                print(log_tag+"unmatched w_names:",w_names)
            print(log_tag,shape_list)
            return
        for l in range(in_len):
            print(log_tag+w_names[l], shape_list[l])


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = tf.constant(float('nan'))
        self.chng = tf.constant(float('nan'))
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if math.isnan(val):
            return
        last_avg = self.avg
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if not math.isnan(last_avg):
            self.chng = self.avg - last_avg


class EarlyStop(object):
    """a class for Early Stopping"""
    def __init__(self,patience,tolerance = 1e-5):
        """

        :param patience: how many step to look back
        :param tolerance: the minimum value to discriminate the two losses
        """
        self.patience = patience
        self.tolerance = tolerance
        self.reset()
        
    def __len__(self):
        return len(self.loss_lst)

    def reset(self):
        self.loss_lst = [np.inf]
        self.no_progress_cnt = 0
        self.status = False

    def last_step(self):
        if len(self.loss_lst)<2:
            return float('nan')
        s0 = self.loss_lst[-2]
        s1 = self.loss_lst[-1]
        if any(np.isinf([s0,s1])):
            return float('nan')
        else:
            out_diff = np.array(s1-s0)
        return np.round(out_diff,4)
    def update(self, loss_val):
        # if current loss is larger than the previous result
        if loss_val > self.loss_lst[-1] or np.abs(loss_val-self.loss_lst[-1]) < self.tolerance:
            self.no_progress_cnt += 1
        else:
            # otherwise, reset the counter
            self.no_progress_cnt = 0

        self.loss_lst.append(loss_val)

        # if the count is larger than the patience, set status to True, break the training process
        if self.no_progress_cnt >= self.patience:
            self.status = True


class score_keeper(object):
    """a class for recording score"""
    def __init__(self):
        self.auc_lst = []
        self.loss_lst = []
    def __len__(self):
        return len(self.auc_lst)
    def update(self, auc_val,loss_val):
        self.auc_lst.append(auc_val)
        self.loss_lst.append(loss_val)
    def last(self):
        "returm the las step's value"
        return self.auc_lst[-1], self.loss_lst[-1]


class timer(object):
    """a class fortime keeper"""
    def __init__(self):
        self.st = 0
        self.interval = 0
        self.n = 0
    def update(self, et):
        self.interval += et - self.st

    def restart(self,st):
        self.st = st
    def avg(self):
        return self.interval/self.n
    def count(self):
        self.n += 1

#
#def set_seed(seed):
#    torch.manual_seed(seed)
#    torch.cuda.manual_seed_all(seed)
#    torch.backends.cudnn.deterministic = True
#    torch.backends.cudnn.benchmark = False
#    np.random.default_rng(5576190)
#    np.random.seed(seed)
#    random.seed(seed)
#    os.environ['PYTHONHASHSEED'] = str(seed)
