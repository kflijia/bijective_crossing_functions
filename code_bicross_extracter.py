# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 09:42:32 2021

@author: Jia
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#regularizer = tf.keras.regularizers.l2(0.1)  # L2 Regularization Penalty
class layer_cross(layers.Layer):
    def __init__(self, setup_inputs, name="layer_cross",**kwargs):
        super(layer_cross, self).__init__(name=name, **kwargs)
        if isinstance(setup_inputs, list): # active
            batch_size, input_dim, init_std = setup_inputs
            self.type = 'active'
            name_prex = "var"
            initializer = keras.initializers.RandomNormal(mean=0., stddev=init_std)
            var_S = tf.Variable(name=name_prex+"_S", initial_value=initializer(shape=(batch_size, input_dim), dtype='float32'), trainable=True)
            var_B = tf.Variable(name=name_prex+"_B", initial_value=initializer(shape=(batch_size, 1), dtype='float32'), trainable=True)
            var_T = tf.Variable(name=name_prex+"_T", initial_value=initializer(shape=(batch_size, input_dim), dtype='float32'), trainable=True)
            var_0 = tf.Variable(name=name_prex+"_0", initial_value=initializer(shape=(batch_size, 1), dtype='float32'), trainable=True)
            self.w_suit = {'w_S':var_S, 'w_B':var_B, 'w_T':var_T, 'w_0':var_0}
        else: # steady
            act_cross = setup_inputs
            batch_size = act_cross.batch_size
            input_dim = act_cross.input_dim
            self.type = 'steady'
            self.w_suit = {}
            for w_tail in ['_S','_B','_T','_0']:
                w = 'w'+w_tail
                self.w_suit[w] = tf.identity(act_cross.w_suit[w], name='std'+w_tail)
        self.batch_size = batch_size
        self.input_dim = input_dim
        return
    def call(self, inputs, run_fwd):
        if run_fwd==True:
            outputs = self.run_forward(inputs[0], inputs[1])
        else:
            outputs = self.run_backward(inputs[0], inputs[1])
        return outputs
    def run_forward(self, inputs_cipher, inputs_text):
        w_S = self.w_suit['w_S']
        w_B = self.w_suit['w_B']
        w_T = self.w_suit['w_T']
        w_0 = self.w_suit['w_0']
        the_cipher = inputs_cipher
        the_text = inputs_text
        # -- insert --
        insert_case_num = 0
        input_len = the_text.shape[0]
        insert_case_num = self.batch_size - input_len
        if insert_case_num > 0: # shorter batch
            insert_zeros = tf.zeros([insert_case_num, self.input_dim])
            the_cipher = tf.concat([the_cipher, insert_zeros], axis=0)
            the_text = tf.concat([the_text, insert_zeros], axis=0)
        multiplier = tf.math.exp(the_cipher * w_S + w_B)
        additer = the_cipher * w_T + w_0
        the_crossed = the_text * multiplier + additer
        if insert_case_num > 0: # shorter batch
            the_crossed = the_crossed[:input_len,:]
        outputs = the_crossed
        return outputs
    def run_backward(self, inputs_crossed, inputs_cipher):
        w_S = self.w_suit['w_S']
        w_B = self.w_suit['w_B']
        w_T = self.w_suit['w_T']
        w_0 = self.w_suit['w_0']
        the_crossed = inputs_crossed
        the_cipher = inputs_cipher
        # -- insert --
        insert_case_num = 0
        input_len = the_crossed.shape[0]
        insert_case_num = self.batch_size - input_len
        if insert_case_num > 0: # shorter batch
            insert_zeros = tf.zeros([insert_case_num, self.input_dim])
            the_cipher = tf.concat([the_cipher, insert_zeros], axis=0)
            the_crossed = tf.concat([the_crossed, insert_zeros], axis=0)
        minuser = the_cipher * w_T + w_0
        divder = tf.math.exp(-(the_cipher * w_S + w_B))
        the_text = (the_crossed - minuser) * divder
        if insert_case_num > 0: # shorter batch
            the_text = the_text[:input_len,:]
        outputs = the_text
        return outputs
    def get_weights(self, in_lists):
        for w in ['w_S','w_B','w_T','w_0']:
            in_lists.append(tf.identity(self.w_suit[w]))
        return in_lists
    def set_weights(self, in_lists):
        # self.type = active, steady, writable
        if self.type == 'steady':
            raise RuntimeError("[Cross] steady: cannot overwrite weights.")
        var_S = tf.Variable(in_lists[0], trainable=True)
        var_B = tf.Variable(in_lists[1], trainable=True)
        var_T = tf.Variable(in_lists[2], trainable=True)
        var_0 = tf.Variable(in_lists[3], trainable=True)
        self.w_suit = {'w_S':var_S, 'w_B':var_B, 'w_T':var_T, 'w_0':var_0}
        return


def new_cross_list(input_dim, batch_size, init_std):
    bundle_size = len(input_dim)
    out_cross_list = []
    for b in range(bundle_size):
        cross_num_b = input_dim[b] - 1
        active_list = []
        for i in range(cross_num_b):
            cross_setup = [batch_size[b], input_dim[b], init_std[b]]
            active_unit = layer_cross(cross_setup, name='act_cr_'+str(i), trainable=True)
            active_unit.trainable = True
            active_list.append(active_unit)
        out_cross_list.append(active_list)
    return out_cross_list
    
def freeze_crosses(in_active_list):
    frozen_list = []
    i = 0
    for L in in_active_list.layers:
        steady_unit = layer_cross(L, name='std_cr_'+str(i), trainable=False)
        steady_unit.trainable=False
        frozen_list.append(steady_unit)
        i += 1
    return frozen_list

def cross_text(in_cipher):
    cross_num = in_cipher.shape[-1] - 1
    text_mat = tf.convert_to_tensor(np.zeros([0, in_cipher.shape[0], in_cipher.shape[1]]))
    text_mat = tf.cast(text_mat, dtype=in_cipher.dtype)
    for i in range(cross_num):
        cipher_part1 = in_cipher[:,:(i+1)]
        cipher_part2 = in_cipher[:,(i+1):]
        plain_arr = tf.concat([cipher_part2, cipher_part1],axis=1)
        text_mat = tf.concat([text_mat, tf.expand_dims(plain_arr,axis=0)], axis=0)
    return text_mat

class Bicrosser(keras.Model):
    def __init__(self, in_crosses, name="bi_crosser", **kwargs):
        super(Bicrosser, self).__init__(name=name, **kwargs)
        self.active_list = in_crosses
        self.cross_num = len(self.active_list)
        self.in_dim = self.cross_num+1
        self.save_dict = {'status':'active'}
    def enable_train(self):
        self.save_dict['status'] = 'active'
        return
    def disable_train(self):
        self.frozen_list = freeze_crosses(self.active_list)
        self.save_dict['status'] = 'steady'
        return
    def call(self, inputs):
        if self.save_dict['status']=='active':
            the_crosses = self.active_list
        else:
            the_crosses = self.frozen_list
        cipher_arr = inputs
        crossed_mat = inputs # (64,10)
        text_mat = tf.convert_to_tensor(np.zeros([0, cipher_arr.shape[0], cipher_arr.shape[1]]))
        text_mat = tf.cast(text_mat, dtype=inputs.dtype)
        i = 0
        for L_fwd in the_crosses: # 9
            cipher_part1 = cipher_arr[:,:(i+1)]
            cipher_part2 = cipher_arr[:,(i+1):]
            plain_arr = tf.concat([cipher_part2, cipher_part1],axis=1)
            text_mat = tf.concat([text_mat, tf.expand_dims(plain_arr,axis=0)], axis=0)
            crossed_arr = L_fwd([cipher_arr, plain_arr], True) # (64,10)
            crossed_mat = tf.concat([crossed_mat, crossed_arr], axis=1)
            i = i+1
            # 64*(10+9*10)
        self.text_mat = text_mat
        return crossed_mat, text_mat
    def get_weights(self, in_list):
        for i in range(self.cross_num):
            in_list.append(self.active_list[i].get_weights([]))
        return in_list
    def set_weights(self, in_weights):
        for i in range(self.cross_num):
            self.active_list[i].set_weights(in_weights[i])
        return
    def get_config(self, in_dict):
        in_dict['cross_num'] = self.cross_num
        in_dict['in_dim'] = self.in_dim
        in_dict['save_dict'] = self.save_dict
        return in_dict


class Bireleaser(keras.Model):
    def __init__(self, in_crosses, name="bi_releaser", **kwargs):
        super(Bireleaser, self).__init__(name=name, **kwargs)
        self.active_list = in_crosses
        self.cross_num = len(self.active_list)
        self.in_dim = self.cross_num+1
        self.save_dict = {'status':'active'}
    def enable_train(self):
        self.save_dict['status'] = 'active'
        return
    def disable_train(self):
        self.frozen_list = freeze_crosses(self.active_list)
        self.save_dict['status'] = 'steady'
        return
    def call(self, inputs_list):
        inputs, in_cipher_arr = inputs_list
        if self.save_dict['status']=='active':
            the_crosses = self.active_list
        else:
            the_crosses = self.frozen_list
        # crossed_mat
        crossed_mat = inputs
        my_cipher_arr = inputs[:,:self.in_dim]
        if len(in_cipher_arr)==0:
            cipher_arr = my_cipher_arr # for testing
        else:
            cipher_arr = in_cipher_arr # for training
        dim_top = self.in_dim
        rls_text_mat = tf.convert_to_tensor(np.zeros([0, cipher_arr.shape[0], cipher_arr.shape[1]]))
        rls_text_mat = tf.cast(rls_text_mat, dtype=inputs.dtype)
        avg_mat = tf.identity(rls_text_mat)
        for i in range(self.cross_num): # 5
            my_dim_end = dim_top + self.in_dim
            crossed_arr = crossed_mat[:,dim_top:my_dim_end]
            dim_top = my_dim_end
            cross_bck = the_crosses[i]
            rls_text_arr = cross_bck([crossed_arr, cipher_arr], False)
            rls_text_mat = tf.concat([rls_text_mat, tf.expand_dims(rls_text_arr,axis=0)],axis=0)
            text_part1 = rls_text_arr[:,:-(i+1)]
            text_part2 = rls_text_arr[:,-(i+1):]
            avg_arr = tf.concat([text_part2, text_part1],axis=1)
            avg_mat = tf.concat([avg_mat, tf.expand_dims(avg_arr,axis=0)],axis=0)
        avg_cipher_arr = tf.reduce_mean(avg_mat,0)
        return my_cipher_arr, avg_cipher_arr, rls_text_mat
    def get_weights(self, in_list):
        for i in range(self.cross_num):
            in_list.append(self.active_list[i].get_weights([]))
        return in_list
    def set_weights(self, in_weights):
        for i in range(self.cross_num):
            self.active_list[i].set_weights(in_weights[i])
        return
    def get_config(self, in_dict):
        in_dict['cross_num'] = self.cross_num
        in_dict['in_dim'] = self.in_dim
        in_dict['save_dict'] = self.save_dict
        return in_dict









