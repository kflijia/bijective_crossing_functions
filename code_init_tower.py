# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:25:55 2021

@author: Jia
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from os.path import exists
#l2=tf.keras.regularizers.l2()
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU
#LeakyReLU = tf.keras.layers.LeakyReLU()
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
import sys
import os
import copy

import code_load_data as dt
import code_bicross_extracter as bi
from code_utils import EarlyStop,AverageMeter

Expd_x_dim = dt.expd_x_dim
Batch_size = dt.batch_size
verb = dt.verb

Dim_state = 16

class Expander(layers.Layer):
    def __init__(self, input_dims, name="expander", **kwargs):
        super(Expander, self).__init__(name=name, **kwargs)
        self.expand_dim = Expd_x_dim
        self.orig_x_dim = input_dims[0]
        self.c_dim = input_dims[1]
        exp = self.expand_dim
        org = self.orig_x_dim
        clone_times = int(exp/org)
        extra_dim = exp - (org * clone_times)
        expd_times_arr = np.zeros((org,)) + clone_times
        extra_clone_idxs = np.random.choice(org, extra_dim, False)
        expd_times_arr[extra_clone_idxs] += 1
        self.expd_times_arr = expd_times_arr
        if org >= exp:
            print("[Expander] No expander needed!")
            self.out_x_dim = org
        else:
            self.out_x_dim = exp
        self.out_dim = self.out_x_dim + self.c_dim
        print("[Expander] dims: orig_x="+str(org)+" expand="+str(exp)+" out="+str(self.out_dim)+". ")
        return
    def call(self, inputs):
        input_x = inputs[:,:self.orig_x_dim]
        input_c = inputs[:,self.orig_x_dim:]
        if not input_c.shape[-1]==self.c_dim:
            raise RuntimeError("[Expander] mismatched input shape!")
        exp = self.expand_dim
        org = self.orig_x_dim
        if org >= exp:
            return inputs
        # -- need expand --
        expd_output = tf.convert_to_tensor(np.zeros((input_x.shape[0],0)))
        for d in range(org):
            clone_n = int(self.expd_times_arr[d])
            d_col = input_x[:,d:(d+1)]
            #clone_cols = tf.constant(d_col.numpy())
            clone_cols = tf.constant(d_col)
            for i in range(clone_n-1):
                clone_cols = tf.concat([clone_cols, d_col], axis=1)
            expd_output = tf.concat([expd_output, clone_cols],axis=1)
        # input_c may be n*1 empty list
        expd_output = tf.concat([expd_output,input_c],axis=1)
        expd_output = tf.cast(expd_output, dtype='float32')
        return expd_output
    def get_config(self, in_dict):
        in_dict['orig_x_dim'] = self.orig_x_dim
        in_dict['c_dim'] = self.c_dim
        in_dict['expand_dim'] = self.expand_dim
        in_dict['out_x_dim'] = self.out_x_dim
        in_dict['out_dim'] = self.out_dim
        in_dict['expd_times_arr'] = self.expd_times_arr
        return in_dict
    def set_config(self, in_dict):
        self.orig_x_dim = in_dict['orig_x_dim']
        self.c_dim = in_dict['c_dim']
        self.expand_dim = in_dict['expand_dim']
        self.out_x_dim = in_dict['out_x_dim']
        self.out_dim = in_dict['out_dim']
        self.expd_times_arr = in_dict['expd_times_arr']
        return

class Shrinker(layers.Layer):
    def __init__(self, in_expander, name="shrinker", **kwargs):
        super(Shrinker, self).__init__(name=name, **kwargs)
        self.expand_dim = in_expander.expand_dim
        self.orig_x_dim = in_expander.orig_x_dim
        self.c_dim = in_expander.c_dim
        self.expd_times_arr = in_expander.expd_times_arr
        self.in_dim = in_expander.out_dim
        return
    def call(self, inputs):
        exp = self.expand_dim
        org = self.orig_x_dim
        if org >= exp:
            return inputs
        input_x = inputs[:, :exp]
        input_c = inputs[:, exp:]
        # input_c may be n*1 empty list
        if not int(input_c.shape[-1])==self.c_dim:
            raise RuntimeError("[Shrinker] mismatched input shape!")
        d_loc = 0
        shrk_output = tf.convert_to_tensor(np.zeros((input_x.shape[0],0)))
        shrk_output = tf.cast(shrk_output, dtype=input_x.dtype)
        for d in range(org):
            d_loc = int(sum(self.expd_times_arr[:d]))
            clone_n = int(self.expd_times_arr[d])
            clone_cols = input_x[:,d_loc:(d_loc+clone_n)]
            out_col = tf.reduce_mean(clone_cols,1)
            out_col = tf.reshape(out_col, [-1,1])
            shrk_output = tf.concat([shrk_output, out_col], axis=1)
        # input_c may be n*1 empty list
        return shrk_output
    def get_config(self, in_dict):
        in_dict['orig_x_dim'] = self.orig_x_dim
        in_dict['c_dim'] = self.c_dim
        in_dict['expand_dim'] = self.expand_dim
        in_dict['in_dim'] = self.in_dim
        in_dict['expd_times_arr'] = self.expd_times_arr
        return in_dict
    def set_config(self, in_dict):
        self.orig_x_dim = in_dict['orig_x_dim']
        self.c_dim = in_dict['c_dim']
        self.expand_dim = in_dict['expand_dim']
        self.in_dim = in_dict['in_dim']
        self.expd_times_arr = in_dict['expd_times_arr']
        return

reg_scale = 1e-10

class Encoder(layers.Layer):
    def __init__(self, auto_dim, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.auto_dim = auto_dim
        self.latent_dim = Dim_state
        layers_units = [64, 48, 32, self.latent_dim]
        layers_list = []
        for us in layers_units:
            batch_norm = layers.BatchNormalization(dtype='float32', trainable=True)
            proj_dense = layers.Dense(us, kernel_regularizer=l2(reg_scale), bias_regularizer=l2(reg_scale), 
                                      kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(),
                                      activity_regularizer=l2(reg_scale), dtype='float32', trainable=True)
            acti_relu = LeakyReLU(alpha=0.05, trainable=True)
            layers_list.append([batch_norm, proj_dense, acti_relu])
        self.layers_units = layers_units
        self.layers_list = layers_list
        return
    def call(self, inputs):
        # 64 * (1+9*10)
        if not int(inputs.shape[-1])==self.auto_dim:
            raise RuntimeError("[Encoder] mismatched input shape!")
        ly_inputs = inputs
        for L in self.layers_list:
            batch_norm = L[0]
            proj_dense = L[1]
            acti_relu = L[2]
            ly_outputs_bn = batch_norm(ly_inputs)
            ly_outputs_proj = proj_dense(ly_outputs_bn)
            ly_outputs_relu = acti_relu(ly_outputs_proj)
            ly_inputs = ly_outputs_relu
        outputs = ly_inputs
        return outputs
    def get_weights(self, in_list):
        for L in self.layers_list:
            in_list.append([L[0].get_weights(), L[1].get_weights(), L[2].get_weights()])
        return in_list
    def set_weights(self, in_list):
        for l,w in enumerate(in_list):
            self.layers_list[l][0].set_weights(w[0])
            self.layers_list[l][1].set_weights(w[1])
            self.layers_list[l][2].set_weights(w[2])
        return
    def get_config(self, in_dict):
        in_dict['auto_dim'] = self.auto_dim
        in_dict['latent_dim'] = self.latent_dim
        in_dict['layers_units'] = self.layers_units
        in_dict['layers_list'] = self.layers_list
        return in_dict


class Decoder(layers.Layer):
    def __init__(self, auto_dim, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.auto_dim = auto_dim
        self.latent_dim = Dim_state
        layers_units = [32, 48, 64, self.auto_dim]
        layers_list = []
        for us in layers_units:
            batch_norm = layers.BatchNormalization(dtype='float32', trainable=True)
            proj_dense = layers.Dense(us, kernel_regularizer=l2(reg_scale), bias_regularizer=l2(reg_scale), 
                                      kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(),
                                      activity_regularizer=l2(reg_scale), dtype='float32', trainable=True)
            acti_relu = LeakyReLU(alpha=0.05, trainable=True)
            layers_list.append([batch_norm, proj_dense, acti_relu])
        #layers_list[-1][2] = layers.Dense(self.auto_dim, activation="sigmoid", dtype='float32')
        layers_list[-1][2] = []
        self.layers_units = layers_units
        self.layers_list = layers_list
        return
    def call(self, inputs):
        # (64, 16)
        if not isinstance(inputs, list): # for place holder
            if not int(inputs.shape[-1])==self.latent_dim:
                raise RuntimeError("[Decoder] mismatched input shape!")
        ly_inputs = inputs
        for L in self.layers_list:
            batch_norm = L[0]
            proj_dense = L[1]
            acti_relu = L[2]
            ly_outputs_bn = batch_norm(ly_inputs)
            ly_outputs_proj = proj_dense(ly_outputs_bn)
            if not acti_relu==[]:
                ly_outputs_relu = acti_relu(ly_outputs_proj)
                ly_inputs = ly_outputs_relu
            else:
                ly_inputs = ly_outputs_proj
        outputs  = ly_inputs
        return outputs
    def get_weights(self, in_list):
        for L in self.layers_list:
            in_list.append([])
            for i in range(3):
                if not L[i]==[]:
                    in_list[-1].append(L[i].get_weights())
                else:
                    in_list[-1].append([])
        return in_list
    def set_weights(self, in_list):
        for l,w in enumerate(in_list):
            for i in range(3):
                if not w[i]==[]:
                    self.layers_list[l][i].set_weights(w[i])
                else:
                    self.layers_list[l][i] = []
        return
    def get_config(self, in_dict):
        in_dict['auto_dim'] = self.auto_dim
        in_dict['latent_dim'] = self.latent_dim
        in_dict['layers_units'] = self.layers_units
        in_dict['layers_list'] = self.layers_list
        return in_dict

class Masker(layers.Layer):
    def __init__(self, mask_dim, name="masker", **kwargs):
        super(Masker, self).__init__(name=name, **kwargs)
        self.mask_dim = mask_dim
        layers_units = [32, 64, self.mask_dim]
        layers_list = []
        for us in layers_units:
            batch_norm = layers.BatchNormalization(dtype='float32', trainable=True)
            proj_dense = layers.Dense(us, kernel_regularizer=l2(1e-10), bias_regularizer=l2(1e-10), 
                                      kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros(),
                                      activity_regularizer=l2(1e-10), dtype='float32', trainable=True)
            acti_relu = LeakyReLU(alpha=0.05, trainable=True)
            layers_list.append([batch_norm, proj_dense, acti_relu])
        # replace output
        #layers_list[-1][2] = layers.Dense(self.mask_dim, activation="sigmoid", dtype='float32')
        layers_list[-1][2] = []
        self.layers_units = layers_units
        self.layers_list = layers_list
        return
    def call(self, in_state):
        inputs = in_state
        for L in self.layers_list:
            batch_norm = L[0]
            proj_dense = L[1]
            acti_relu = L[2]
            ly_outputs_bn = batch_norm(inputs)
            ly_outputs_proj = proj_dense(ly_outputs_bn)
            if not acti_relu==[]:
                ly_outputs_relu = acti_relu(ly_outputs_proj)
                inputs = ly_outputs_relu
            else:
                inputs = ly_outputs_proj
        outputs  = inputs
        return outputs
    def get_weights(self, in_list):
        for L in self.layers_list:
            in_list.append([])
            for i in range(3):
                if not L[i]==[]:
                    in_list[-1].append(L[i].get_weights())
                else:
                    in_list[-1].append([])
        return in_list
    def set_weights(self, in_list):
        for l,w in enumerate(in_list):
            for i in range(3):
                if not w[i]==[]:
                    self.layers_list[l][i].set_weights(w[i])
                else:
                    self.layers_list[l][i] = []
        return
    def get_config(self, in_dict):
        in_dict['mask_dim'] = self.mask_dim
        in_dict['layers_units'] = self.layers_units
        in_dict['layers_list'] = self.layers_list
        return in_dict

class Bundler(object):
    def __init__(self, in_set_up):
        super(Bundler, self).__init__()
        in_obj, in_setup_list = in_set_up
        b_size = len(in_setup_list)
        self.obj_list = [in_obj(in_setup_list[b]) for b in range(b_size)]
        self.config_list = [self.obj_list[b].get_config({}) for b in range(b_size)]
        self.config = {}
        conf_dict = self.config_list[0]
        for c in conf_dict:
            self.config[c] = [self.config_list[b][c] for b in range(b_size)]
        self.bundle_size = b_size
    def __call__(self, inputs):
        outputs_list = []    
        for i in range(self.bundle_size):
            outputs_list.append(self.obj_list[i](inputs[i]))
        return outputs_list
    def get_weights(self, in_lists):
        for i in range(self.bundle_size):
            in_lists.append(self.obj_list[i].get_weights([]))
        return in_lists
    def set_weights(self, in_weights):
        for i in range(self.bundle_size):
            self.obj_list[i].set_weights(in_weights[i])
        return
    def get_config(self, in_lists):
        for i in range(self.bundle_size):
            in_lists.append(self.obj_list[i].get_config({}))
        return in_lists
    def set_config(self, in_confs):
        for i in range(self.bundle_size):
            self.obj_list[i].set_config(in_confs[i])
        return
    def do_fun(self, in_fun_str):
        if "disable" in in_fun_str:
            for obj in self.obj_list:
                obj.disable_train()
        if "enable" in in_fun_str:
            for obj in self.obj_list:
                obj.enable_train()
        return

class Tower_Base(layers.Layer):
    def __init__(self, input_dims_bundle, name="tower_base", **kwargs):
        super(Tower_Base, self).__init__(name=name, **kwargs)
        self.expander = Bundler([Expander, input_dims_bundle])
        expd_dim_bundle = self.expander.config['out_dim']
        auto_dim_bundle = [expd_dim**2 for expd_dim in expd_dim_bundle]
        self.encoder = Bundler([Encoder, auto_dim_bundle])
        self.extracter = [] # create in Tower
        self.bundle_size = len(input_dims_bundle)
    def call(self, input_x, input_c):
        b_size = self.bundle_size
        inputs_mat_bundle = []
        for b in range(b_size):
            inputs_mat = np.concatenate([input_x[b], input_c[b]], axis=1)
            inputs_mat_bundle.append(inputs_mat)
        inputs = inputs_mat_bundle
        input_x_merge = self.merge_output(input_x)
        expd_output = self.expander(inputs)
        extr_output = self.extracter(expd_output)
        extr_crossed = [extr_output[b][0] for b in range(b_size)]
        extr_text_mat = [extr_output[b][1] for b in range(b_size)]
        extr_output = [extr_crossed, extr_text_mat]
        state_output = self.encoder(extr_crossed)
        x_dim = self.expander.config['orig_x_dim']
        c_dim = self.expander.config['c_dim']
        expd_x_dim = self.expander.config['out_x_dim']
        expd_mask = []
        c_patch = []
        output_dtype = state_output[0].dtype
        for b in range(b_size):
            expd_output[b] = tf.cast(expd_output[b], dtype=output_dtype)
            # --- for mask ---
            input_x_b = tf.convert_to_tensor(inputs[b][:,:(x_dim[b])]) # inputs is np.array
            expd_x_output_b = expd_output[b][:,:(expd_x_dim[b])]
            if "strided_slice" in str(input_x_b[0]): # place holder!
                expd_x_mask_b = tf.ones_like(expd_x_output_b)
            else:
                expd_x_mask_b = abs(expd_x_output_b.numpy())>0.005
            c_cover_b = tf.zeros(shape=[input_x_b.shape[0], c_dim[b]])
            expd_mask_b = tf.concat([expd_x_mask_b, c_cover_b], axis=1)
            expd_mask_b = tf.cast(expd_mask_b, dtype=output_dtype)
            expd_mask.append(expd_mask_b)
            # --- for training ---
            c_patch_b =  expd_output[b][:,(-c_dim[b]):]
            c_patch.append(c_patch_b)
        return [expd_output, extr_output, state_output, expd_mask, c_patch, input_x_merge]
    def merge_output(self, x_bundle):
        b_size = self.bundle_size
        if b_size==1:
            x_out = x_bundle[0]
            x_out = tf.cast(x_out, dtype='float32')
            return x_out
        x_len = np.shape(x_bundle[0])[0]
        x_merged = tf.constant([[] for i in range(x_len)], dtype='float32')
        for b in range(b_size):
            x_b = tf.cast(x_bundle[b], dtype='float32')
            x_merged = tf.concat([x_merged, x_b], axis=1)
        return x_merged
    def get_weights(self, in_list):
        in_list.append(self.extracter.get_weights([]))
        in_list.append(self.encoder.get_weights([]))
        in_list.append(self.expander.get_config([])) # use dict in bundler
        return in_list
    def set_weights(self, in_list):
        cross_weights, encd_weights, expd_confs = in_list
        self.extracter.set_weights(cross_weights)
        self.encoder.set_weights(encd_weights)
        self.expander.set_config(expd_confs)
        return

class Tower_Top(layers.Layer):
    def __init__(self, tower_base, name="tower_top", **kwargs):
        super(Tower_Top, self).__init__(name=name, **kwargs)
        self.c_dim = tower_base.expander.config['c_dim']
        expd_x_dim = tower_base.expander.config['out_x_dim']
        auto_dim = tower_base.encoder.config['auto_dim']
        self.decoder = Bundler([Decoder, auto_dim])
        self.shrinker = Bundler([Shrinker, tower_base.expander.obj_list])
        self.masker = Bundler([Masker, expd_x_dim])
        self.releaser = [] # create in Tower
        self.bundle_size = tower_base.bundle_size
        return
    def call(self, in_state, in_c_patch, in_expd=[], in_mask=[]):
        b_size = self.bundle_size
        if in_expd==[]:
            in_expd = [[] for b in range(b_size)]
            in_mask = [[] for b in range(b_size)]
        decd_output = self.decoder(in_state) # crss
        rls_input = [[decd_output[b], in_expd[b]] for b in range(b_size)]
        rls_output = self.releaser(rls_input) # expd
        mask_expd_x = self.masker(in_state)
        # --- c_cover & c_patch ---
        rls_cipher = []
        rls_cipher_pred = []
        rls_avg_cipher = []
        rls_text_mat = []
        mask_expd = []
        mask_bi_expd = []
        output_dtype =  rls_output[0][0].dtype
        for b in range(b_size):
            rls_cipher_b = rls_output[b][0]
            rls_avg_cipher_b = rls_output[b][1]
            rls_text_mat_b = rls_output[b][2]
            # c_patch
            c_patch_b = in_c_patch[b]
            c_dim_b = int(c_patch_b.shape[-1])
            rls_cipher_b = tf.concat([rls_cipher_b[:,:-c_dim_b], c_patch_b], axis=1)
            rls_avg_cipher_b = tf.concat([rls_avg_cipher_b[:,:-c_dim_b], c_patch_b], axis=1)
            # c_cover
            c_cover_b = tf.zeros_like(c_patch_b)
            mask_expd_b = tf.concat([mask_expd_x[b], c_cover_b], axis=1)
            # expd_mask
            mask_bi_expd_b = tf.ones_like(mask_expd_b) # default
            if not "strided_slice" in str(in_state[b][0]): # not place holder!
                mask_bi_expd_b = tf.cast(mask_expd_b.numpy()>=0.5, dtype=output_dtype)
            the_mask_b = in_mask[b]
            if len(the_mask_b)==0:
                the_mask_b = mask_bi_expd_b
            rls_cipher_pred_b = tf.multiply(rls_cipher_b, the_mask_b)
            # bundle
            rls_cipher.append(rls_cipher_b)
            rls_cipher_pred.append(rls_cipher_pred_b)
            rls_avg_cipher.append(rls_avg_cipher_b)
            rls_text_mat.append(rls_text_mat_b)  # no c_patch for rls_text_mat
            mask_expd.append(mask_expd_b)
            mask_bi_expd.append(mask_bi_expd_b)
        rls_output = [rls_cipher, rls_avg_cipher, rls_text_mat, rls_cipher_pred]
        mask_out_pair = [mask_expd, mask_bi_expd]
        # merge mask        
        mask_recv_x = self.shrinker(mask_expd)
        mask_recv_x_merge = self.merge_output(mask_recv_x)
        mask_bi_recv_x_merge = tf.cast(mask_recv_x_merge.numpy()>=0.5, dtype=output_dtype)
        mask_merge_pair = [mask_recv_x_merge, mask_bi_recv_x_merge]
        # output
        shrk_recv_x = self.shrinker(rls_cipher) # recv
        shrk_recv_x_pred = self.shrinker(rls_cipher_pred) # recv_pred
        shrk_out_pair = [shrk_recv_x, shrk_recv_x_pred]
        shrk_merge_pair = [self.merge_output(shrk_recv_x), self.merge_output(shrk_recv_x_pred)]
        return [decd_output, rls_output, shrk_out_pair, shrk_merge_pair, mask_out_pair, mask_merge_pair]
    def merge_output(self, x_bundle):
        b_size = self.bundle_size
        if b_size==1:
            x_out = x_bundle[0]
            x_out = tf.cast(x_out, dtype='float32')
            return x_out
        x_len = np.shape(x_bundle[0])[0]
        x_merged = tf.constant([[] for i in range(x_len)], dtype='float32')
        for b in range(b_size):
            x_merged = tf.concat([x_merged, x_bundle[b]], axis=1)
        return x_merged
    def get_weights(self, in_list):
        cross_weights = self.releaser.get_weights([])
        in_list.append(cross_weights)
        decd_weights = self.decoder.get_weights([])
        in_list.append(decd_weights)
        mskr_weights = self.masker.get_weights([])
        in_list.append(mskr_weights)
        shrk_confs = self.shrinker.get_config([])
        in_list.append(shrk_confs) # use dict in bundler
        return in_list
    def check_crosses(self, in_cross_weights):
        if in_cross_weights==[]:
            return
        rls_cross_weights = self.releaser.get_weights([])
        for b in range(self.bundle_size):
            in_w_b = in_cross_weights[b]
            rls_w_b = rls_cross_weights[b]
            check_tags = []
            for i, w in enumerate(in_w_b):
                v = rls_w_b[i] # set by crosser
                cr_check_arr = []
                for j in range(len(w)):
                    check_mat = w[j].numpy()==v[j].numpy()
                    cr_check_arr.append(all([all(k) for k in check_mat]))
                cr_check_tag = all(cr_check_arr)
                check_tags.append(cr_check_tag)
            if not all(check_tags):
                raise RuntimeError("[Tower Top] bundle #"+str(b)+" inconsistent releaser weights setting!")    
        return
    def set_weights(self, in_list, cross_w_extr=[]):
        self.check_crosses(cross_w_extr)
        # checking, to confirm using the same active_list
        if len(in_list)==3:
            decd_weights, mskr_weights, shrk_confs = in_list
        else:
            cross_weights, decd_weights, mskr_weights, shrk_confs = in_list
            self.releaser.set_weights(cross_weights)
        self.decoder.set_weights(decd_weights)
        self.masker.set_weights(mskr_weights)
        self.shrinker.set_config(shrk_confs)
        return
 

 #----------------------------------------------------
 #   
bce_loss_fn = tf.keras.losses.BinaryCrossentropy()

mse_loss_fn = tf.keras.losses.MeanSquaredError()

def rmse_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    mse = tf.reduce_mean(squared_difference, axis=-1)
    rmse = tf.reduce_mean(tf.sqrt(mse), axis=-1)
    #mse = (np.square(y_true - y_pred)).mean(axis=1)
    #rmse = np.mean(np.sqrt(mse))
    return rmse

def both_loss_fn(y_true, y_pred):
    if len(y_true.shape)<=2: # 2D dimension
        mse = mse_loss_fn(y_true, y_pred)
        rmse = rmse_loss_fn(y_true, y_pred)
        return mse, rmse
    else: # 3D dimension
        mse_arr = tf.convert_to_tensor([mse_loss_fn(y_true[0], y_pred[0])])
        rmse_arr = tf.convert_to_tensor([mse_loss_fn(y_true[0], y_pred[0])])
        i_num = int(y_true.shape[0])
        for i in range(1, i_num):
            mse_arr = tf.concat([mse_arr, [mse_loss_fn(y_true[i], y_pred[i])]], axis=0)
            rmse_arr = tf.concat([rmse_arr, [rmse_loss_fn(y_true[i], y_pred[i])]], axis=0)
        return mse_arr, rmse_arr


bicross_std_disc = {"j":0.1, "f":0.01, "i":0.1, "a":0.1, "b":0.1} # "df":0.1, "cde":0.1, "ghi":0.1, 
stopper_toler_disc = {"j":0.001, "f":0.0002, "h":0.0005, "i":0.001, "a":0.0005, "b":0.0005, "e":0.0005} # "cde":0.001, "ghi":0.001, 
class Tower(keras.Model):
    def __init__(self, in_keys, in_data_obj):
        super(Tower, self).__init__()
        if isinstance(in_keys, str):
            in_keys = [char for char in in_keys]
        self.keys = in_keys
        self.d_obj = in_data_obj
        # --- base + top ---
        b_size = len(self.keys)
        in_dims = [self.d_obj.get_dims(k) for k in self.keys]
        self.tw_base = Tower_Base(in_dims)
        self.tw_top = Tower_Top(self.tw_base)
        # --- crosser + releaser ---
        exp_dim_bundle = self.tw_base.expander.config['out_dim']
        batch_size_bundle = [Batch_size for b in range(b_size)]
        init_std = 0.02
        def_tolr = 0.001
        std_bundle = []
        tolr_bundle = []
        for k in self.keys:
            if k in bicross_std_disc:
                std_bundle.append(bicross_std_disc[k])
            else:
                std_bundle.append(init_std)
            if k in stopper_toler_disc:
                tolr_bundle.append(stopper_toler_disc[k])
            else:
                tolr_bundle.append(def_tolr)
        self.stopper_toler = min(tolr_bundle)
        self.cross_list_bundle = bi.new_cross_list(exp_dim_bundle, batch_size_bundle, std_bundle)
        self.tw_base.extracter = Bundler([bi.Bicrosser, self.cross_list_bundle])
        self.tw_top.releaser = Bundler([bi.Bireleaser, self.cross_list_bundle])
        # --- for merge ---
        tw_colnames_bundle = []
        for k in self.keys:
            tw_colnames_bundle.append(self.d_obj.get_names(k)[0])
        # ---
        self.saved_fname, output_fname = dt.get_discv_tower_fnames(self.keys)
        self.output_fname_orig = output_fname+'_orig.csv'
        self.output_fname_recv = output_fname+'_recv.csv'
        self.finished_tag = False # training is only for single-key tower
        self.finished_tag_bundle = [False for b in range(b_size)]
        self.loss_recorder = {}
        # instatalizte
        x_seed_bundle, c_seed_bundle = self.d_obj.get_seeds_bundle(self.keys, Batch_size)
        _ = self.call(x_seed_bundle, c_seed_bundle)
        # --- gradients models: single-key using only! ---
        # <<model_auto>> autoencoder = encoder + decoder
        # expd is for cipher-part, crss is for full = cipher-part + avg-part
        auto_dim = self.tw_base.encoder.config['auto_dim']
        input_auto = layers.Input(shape=(auto_dim[0],), batch_size=Batch_size, dtype='float32')
        #for b in range(b_size):
        #    input_auto.append(layers.Input(shape=(auto_dim[b],), batch_size=Batch_size, dtype='float32'))
        state_0 = self.tw_base.encoder.obj_list[0](input_auto)
        x_auto = self.tw_top.decoder.obj_list[0](state_0)
        self.model_auto = keras.Model(input_auto, x_auto)
        # <<model_cross>> crosser ( + autoencoder if auto trainable)
        # text_mat-part
        # input_2 = []
        # for b in range(b_size):
        #     input_2.append(layers.Input(shape=(auto_dim[b],), batch_size=Batch_size, dtype='float32'))
        # rls_out = self.tw_top.releaser(input_2)
        # self.model_cross = [keras.Model(input_2[b], rls_out[b]) for b in range(b_size)]
        self.model_cross = self.tw_top.releaser.obj_list[0]
        # <<model_masker>> masker only
        #input_state = [layers.Input(shape=(Dim_state,), batch_size=Batch_size, dtype='float32') for b in range(b_size)]
        input_state = layers.Input(shape=(Dim_state,), batch_size=Batch_size, dtype='float32')
        x_mask_0 = self.tw_top.masker.obj_list[0](input_state)
        self.model_masker = keras.Model(input_state, x_mask_0)
        return
    def copy_top(self, cp_top, tail_str=''):
        cp_cross_list = []
        for cr_list in self.cross_list_bundle:
            cp_cr_list = bi.freeze_crosses(cr_list)
            for cr in cp_cr_list:
                cr.type = "writable" # make writable, in case used in unit
            cp_cross_list.append(cp_cr_list)
        cp_top.releaser = Bundler([bi.Bireleaser, cp_cross_list])
        x_seed_bundle, c_seed_bundle = self.d_obj.get_seeds_bundle(self.keys, Batch_size)
        base_out = self.call_base(x_seed_bundle, c_seed_bundle)
        expd_output, _, state_output, in_mask_expd, c_patch, _ = base_out
        _ = cp_top(state_output, c_patch, expd_output, in_mask_expd)
        cp_top.decoder.set_weights(self.tw_top.decoder.get_weights([]))
        cp_top.shrinker.set_config(self.tw_top.shrinker.get_config([]))
        if len(tail_str)>0:
            for b in range(len(self.keys)):
                cp_top.decoder.obj_list[b]._name = "de"+tail_str+str(b)
                cp_top.masker.obj_list[b]._name = "msk"+tail_str+str(b)
        return cp_top
    def call_base(self, input_x, input_c):
        base_out = self.tw_base(input_x,input_c)
        return base_out
    def call(self, input_x, input_c, training=True):
        base_out = self.call_base(input_x, input_c)
        expd_output, extr_output, state_output, in_mask_expd, c_patch, input_x_merge = base_out
        if training:
            top_out = self.tw_top(state_output, c_patch, expd_output, in_mask_expd)
        else:
            top_out = self.tw_top(state_output, c_patch)
        decd_output, rls_output, shrk_out_pair, shrk_merge_pair, mask_out_pair, mask_merge_pair = top_out
        x_recv, x_recv_pred = shrk_out_pair
        x_merge, x_merge_pred = shrk_merge_pair
        mask_expd, mask_expd_bi = mask_out_pair
        mask_merge_x, mask_merge_x_bi = mask_merge_pair
        # when training, x_recv_pred is true mask applied
        # when testing, x_recv_pred is pred mask applied (== x_recv*mask_recv_x_bi)
        x_orig = self.descl_output(x_recv_pred)
        x_orig_merge = self.tw_top.merge_output(x_orig)
        shrk_merge_triple = [x_merge, x_merge_pred, x_orig_merge]
        if training:
            base_out = [expd_output, extr_output, state_output, in_mask_expd, input_x_merge]
            top_out = [decd_output, rls_output, mask_out_pair, shrk_merge_triple]
            return [base_out, top_out]
        else:
            the_predict = tf.multiply(x_merge, mask_merge_x_bi)
            # x_merge * mask_merge_bi is more accurate than x_merge_pred
            return the_predict
    def descl_output(self, recv_x_bundle):
        b_size = len(self.keys)
        orig_x_bundle = []
        for b in range(b_size):
            b_key = self.keys[b]
            b_x = recv_x_bundle[b]
            b_orig_x = self.d_obj.descale_x(b_key, b_x)
            orig_x_bundle.append(b_orig_x)
        return orig_x_bundle
    def convert_single(self, in_bundle_list):
        out_single_list = []
        for the_bundle in in_bundle_list:
            assert len(the_bundle)==1
            out_single_list.append(the_bundle[0])
        return out_single_list
    def convert_bundle(self, in_bundle_list):
        merged_list = []
        for the_bundle in in_bundle_list:
            initial_shape = list(np.shape(the_bundle[0]))
            initial_shape[-1] = 0
            the_merged_data = tf.convert_to_tensor(np.zeros(initial_shape), dtype='float32')
            for the_data in the_bundle:
                the_merged_data = tf.concat([the_merged_data, the_data], axis=-1)
            merged_list.append(the_merged_data)
        return merged_list
    def output_reset(self):
        tw_colnames = self.d_obj.get_names(self.keys)[0]
        orig_header = []
        recv_header = []
        for h in ['OT','OP']:
            orig_header.extend([h+'_'+c for c in tw_colnames])
        for h in ['RT','RP']:
            recv_header.extend([h+'_'+c for c in tw_colnames])
        init_stdout = sys.stdout
        with open(self.output_fname_orig, 'w') as f_orig:
            sys.stdout = f_orig
            for h in orig_header:
                print("{0: <5} ".format(h), end='')
            print("")
        with open(self.output_fname_recv, 'w') as f_recv:
            sys.stdout = f_recv
            for h in recv_header:
                print("{0: <5} ".format(h), end='')
            print("")
        sys.stdout = init_stdout
        pd.options.display.float_format = '{:,.4f}'.format
        return
    def output_file(self, recv_in_x, recv_out_x, orig_in_x, orig_out_x):
        init_stdout = sys.stdout
        # -- recv --
        recv_data = tf.concat([recv_in_x, recv_out_x], axis=1)
        recv_df = pd.DataFrame(recv_data.numpy())
        with open(self.output_fname_recv, 'a+') as f_recv:
            sys.stdout = f_recv
            print(recv_df.to_string(index=False, header=False))
        # -- orig --
        orig_data = tf.concat([orig_in_x, orig_out_x], axis=1)        
        orig_df = pd.DataFrame(orig_data.numpy())
        with open(self.output_fname_orig, 'a+') as f_orig:
            sys.stdout = f_orig
            print(orig_df.to_string(index=False, header=False))
        sys.stdout = init_stdout
        return
    def output_backup(self):
        ln_command = "mv "+self.output_fname_recv+" "+self.output_fname_recv+"_bk"
        os.system(ln_command)
        ln_command = "mv "+self.output_fname_orig+" "+self.output_fname_orig+"_bk"
        os.system(ln_command)
        return
    def output_recover(self):
        ln_command = "mv "+self.output_fname_recv+"_bk "+self.output_fname_recv
        os.system(ln_command)
        ln_command = "mv "+self.output_fname_orig+"_bk "+self.output_fname_orig
        os.system(ln_command)
        return
    def save(self):
        train_params = [self.finished_tag, self.loss_recorder]
        if len(self.keys)==1:
            cross_weights, encd_weights, expd_confs = self.tw_base.get_weights([])
            _, decd_weights, mskr_weights, shrk_confs = self.tw_top.get_weights([])
            saved_list = [cross_weights[0], encd_weights[0], expd_confs[0], decd_weights[0], mskr_weights[0], shrk_confs[0], train_params]
        else:
            saved_list = [[],[],[],[],[],[], train_params]
        dt.save_obj(saved_list, self.saved_fname)
        return
    def load(self):
        if exists(self.saved_fname):
            loaded_list = dt.load_obj(self.saved_fname)
            _, _, _, _, _, _, train_params = loaded_list
            self.finished_tag = train_params[0]
            self.loss_recorder = train_params[1]
        if len(self.keys)==1:
            if 'loaded_list' in locals():
                cross_weights, encd_weights, expd_confs, decd_weights, mskr_weights, shrk_confs, _ = loaded_list
                cross_weights = [cross_weights]
                encd_weights = [encd_weights]
                expd_confs = [expd_confs]
                decd_weights = [decd_weights]
                mskr_weights = [mskr_weights]
                shrk_confs = [shrk_confs]
            else:
                if exists(self.saved_fname):
                    raise RuntimeError("[Tower] loading error: file ", self.saved_fname)
                return
        else:
            cross_weights = [[] for k in self.keys]
            encd_weights = [[] for k in self.keys]
            expd_confs = [[] for k in self.keys]
            decd_weights = [[] for k in self.keys]
            mskr_weights = [[] for k in self.keys]
            shrk_confs = [[] for k in self.keys]
            finished_tag_bundle = [False for k in self.keys]
            b = 0
            for k in self.keys:
                k_saved_fname = dt.get_discv_tower_fnames(k)[0]
                if not exists(k_saved_fname):
                    keys_str = ''.join(self.keys)
                    raise RuntimeError("[Tower "+keys_str+"] requires "+k_saved_fname+".")
                loaded_list = dt.load_obj(k_saved_fname)
                cross_weights[b], encd_weights[b], expd_confs[b], decd_weights[b], mskr_weights[b], shrk_confs[b], [finished_tag_bundle[b],_] = loaded_list
                b += 1
            self.finished_tag = all(finished_tag_bundle)
            self.finished_tag_bundle = finished_tag_bundle
        self.tw_base.set_weights([cross_weights, encd_weights, expd_confs]) # set cross_weights
        self.tw_top.set_weights([decd_weights, mskr_weights, shrk_confs], cross_weights) # check cross_weights
        return
    def do_train(self, n_shuff=6, n_epoch_rounds=[4,4], output_only=False, enforce=False):
        keys_str = ''.join(self.keys)
        loss_rd = self.loss_recorder
        stages = ['crss','rls','expd','recv','orig']
        hist_epoches = 0
        n_eps = 0
        lowest_recv_rmse = float('inf')
        if len(loss_rd)>0:
            hist_epoches = int(loss_rd['other'][1])
            n_eps = int(loss_rd['other'][2])
            print("["+keys_str+"] history Epoch %d, last n_epoch %d" % (hist_epoches, n_eps))
            for a in stages:
                print("["+keys_str+"] history mean "+a+"[mse,rmse]={}".format(loss_rd[a]))
            print("["+keys_str+"] history mean mask={}".format(loss_rd['other'][0]))
            lowest_recv_rmse = np.copy(loss_rd['recv'][1])
        # --- active for single-key tower only!!! ---
        b_size = len(self.keys)       
        if b_size==1: # single-key
            bundle_tag = False
            my_convert_fun = self.convert_single
            if self.finished_tag and not output_only:
                print("[Tower "+keys_str+"] training finished already.")
                return
        else: # multi-keys
            bundle_tag = True
            output_only = True
            my_convert_fun = self.convert_bundle
            print("[Tower "+keys_str+"] no training for multi-keys tower. Output Only.")
            if not self.finished_tag:
                unfinished_keys = list(np.array(self.keys)[np.invert(self.finished_tag_bundle)])
                print("      unfinished single-key tower:", unfinished_keys)
            else:
                print("      all single-key tower finished.")
        self.n_shuff = n_shuff
        self.n_epoch_rounds = n_epoch_rounds
        n_epoch_1st = n_epoch_rounds[0]
        n_epoch_2nd = n_epoch_rounds[1]
        n_epoch = sum(n_epoch_rounds)            
        if output_only:
            fixed_finished_tag = np.copy(self.finished_tag)
            self.n_shuff = 1
            n_epoch = 1            
            print("******** "+keys_str+" Tower output batch_size=%d ********" % (Batch_size))
        else:
            print("******** "+keys_str+" Tower no-shuffle training batch_size=%d ********" % (Batch_size))
        # --- prepare ---
        loss_rd = {}
        meaners_MSE = {}
        meaners_RMSE = {}
        meaner_mask = AverageMeter()
        for a in stages:
            meaners_MSE[a] = AverageMeter()
            meaners_RMSE[a] = AverageMeter()
        stoper = EarlyStop(patience=4, tolerance=self.stopper_toler)
        optimizer_1 = Adam(learning_rate=0.001, clipnorm=1.0, clipvalue=0.5)
        optimizer_2 = Adam(learning_rate=0.001, clipnorm=1.0, clipvalue=0.5)
        optimizer_3 = Adam(learning_rate=0.001, clipnorm=1.0, clipvalue=0.5)
        optimizer_mask = Adam(learning_rate=0.001, clipnorm=1.0, clipvalue=0.5)
        n_eps = 0
        # --- data bundle ---
        x_train_bundle, c_train_bundle = self.d_obj.get_train_bundle(self.keys)
        dataset_tuple = tuple(x_train_bundle + c_train_bundle)
        # --- start ---
        for sh in range(self.n_shuff):
            if stoper.status:
                break
            # Round 1st: <<model_auto>> with expd, <<model_cross>> with crosser trainable
            # Round 2nd: <<model_auto>> with crss, <<model_cross>>
            for ep in range(n_epoch):
                Epoche = n_epoch * sh + ep + 1
                if stoper.status:
                    break
                self.tw_top.releaser.do_fun("enable")# in Cross gradient 
                if ep < n_epoch_1st: # Round 1st Auto
                    self.tw_base.extracter.do_fun("disable") # no gradient, but update
                    round_str = "Auto"
                    round_ep = ep
                else: # Round 2nd Cross
                    round_str = "Cross"
                    round_ep = ep - n_epoch_1st
                print("[Tower]["+keys_str+"] Epoche %d [iterate %d] [round %s] [epoche %d] " % (Epoche,sh+1,round_str,round_ep+1))
                self.output_reset()
                meaner_mask.reset()
                for a in stages:
                    meaners_MSE[a].reset()
                    meaners_RMSE[a].reset()
                state_clips = [float('inf'), -float('inf')] # [min, max]
                dataset_train = tf.data.Dataset.from_tensor_slices(dataset_tuple)
                dataset_train = dataset_train.batch(Batch_size)
                for step, batch_train in enumerate(dataset_train):
                    x_batch_bundle = batch_train[:b_size]
                    c_batch_bundle = batch_train[-b_size:]
                    with tf.GradientTape(persistent=True) as tape:
                        base_out, top_out = self.call(x_batch_bundle, c_batch_bundle)
                        [expd_in, [crss_in, text_mat_in], tw_state, mask_in, recv_in_x] = base_out
                        [crss_out, rls_out, mask_out_pair, shrk_out_triple] = top_out
                        rls_cipher, rls_avg, rls_text_mat, rls_cipher_metric = rls_out
                        mask_out, mask_out_binary = mask_out_pair
                        expd_out = rls_cipher # or rls_avg / rls_cipher_metric
                        text_mat_out = rls_text_mat
                        # convert to single
                        convert_inner_list = [crss_in, crss_out, expd_in, expd_out, text_mat_in, text_mat_out]
                        [crss_in, crss_out, expd_in, expd_out, text_mat_in, text_mat_out] = my_convert_fun(convert_inner_list)
                        convert_output_list = [tw_state, mask_in, mask_out, mask_out_binary]
                        [tw_state, mask_in, mask_out, mask_out_binary] = my_convert_fun(convert_output_list)
                        crss_mse, crss_rmse = both_loss_fn(crss_in, crss_out)
                        expd_mse, expd_rmse = both_loss_fn(expd_in, expd_out)
                        rls_mse_arr, rls_rmse_arr = both_loss_fn(text_mat_in, text_mat_out) # 3D
                        mask_bce = bce_loss_fn(mask_in, mask_out)
                        # ---
                        loss_Auto_1 = expd_rmse
                        loss_Auto_2 = crss_rmse
                        loss_Cross = rls_rmse_arr
                        loss_Mask = mask_bce
                    if not output_only: # single-key only
                        if round_str=="Auto":
                            grads_auto_1 = tape.gradient(loss_Auto_1, self.model_auto.trainable_weights)
                            _ = optimizer_1.apply_gradients(zip(grads_auto_1, self.model_auto.trainable_weights))
                            grads_auto_2 = tape.gradient(loss_Auto_2, self.model_auto.trainable_weights)
                            _ = optimizer_2.apply_gradients(zip(grads_auto_2, self.model_auto.trainable_weights))
                            grad_mask = tape.gradient(loss_Mask, self.model_masker.trainable_weights)
                            _ = optimizer_mask.apply_gradients(zip(grad_mask, self.model_masker.trainable_weights))
                        else:
                            grads_cross = tape.gradient(loss_Cross, self.model_cross.trainable_weights)
                            _ = optimizer_3.apply_gradients(zip(grads_cross, self.model_cross.trainable_weights))
                    # --- metric evaluations ---
                    recv_out_x, recv_out_x_metric, orig_out_x_metric = shrk_out_triple
                    rls_mse = tf.convert_to_tensor(np.mean(rls_mse_arr))
                    rls_rmse = tf.convert_to_tensor(np.mean(rls_rmse_arr))
                    orig_in_x_bundle, _ = self.d_obj.get_batch_bundle_orig(self.keys, step, Batch_size)
                    orig_in_x = self.tw_top.merge_output(orig_in_x_bundle)
                    recv_mse, recv_rmse = both_loss_fn(recv_in_x, recv_out_x_metric)
                    orig_mse, orig_rmse = both_loss_fn(orig_in_x, orig_out_x_metric)
                    mask_binary_bce = bce_loss_fn(mask_in, mask_out_binary)
                    meaners_MSE['crss'].update(crss_mse)
                    meaners_RMSE['crss'].update(crss_rmse)
                    meaners_MSE['rls'].update(rls_mse)
                    meaners_RMSE['rls'].update(rls_rmse)
                    meaners_MSE['expd'].update(expd_mse)
                    meaners_RMSE['expd'].update(expd_rmse)
                    meaners_MSE['recv'].update(recv_mse)
                    meaners_RMSE['recv'].update(recv_rmse)
                    meaners_MSE['orig'].update(orig_mse)
                    meaners_RMSE['orig'].update(orig_rmse)
                    meaner_mask.update(mask_binary_bce)
                    # --- log display ---
                    state_min = np.min(tw_state)
                    state_max = np.max(tw_state)
                    state_mean = np.mean(tw_state)
                    state_clips[0] = min(state_clips[0], state_min)
                    state_clips[1] = max(state_clips[1], state_max)
                    if step % 100 == 0 and verb >= 3:
                        # arr_crss = [np.round(loss,3) for loss in [crss_mse, crss_rmse]]
                        # arr_expd = [np.round(loss,3) for loss in [expd_mse, expd_rmse]]
                        arr_recv = [np.round(loss,3) for loss in [recv_mse, recv_rmse]]
                        arr_orig = [np.round(loss,3) for loss in [orig_mse, orig_rmse]]
                        print("    [Tower]["+keys_str+"]....batches "+str(step)+": clip{%.3f,%.3f}%.1f" % 
                              (state_min, state_max, state_mean),end='')
                        print("%"+" < Auto_1 = %.3f, Auto_2 = %.3f, Mask = %.3f >  < Cross = %.3f >." % (loss_Auto_1, loss_Auto_2, mask_binary_bce, rls_rmse))
                        # print("                   [mse,rmse] crss={}.".format(arr_crss), end='' )
                        # print(" expd_x={}.".format(arr_expd), end='' )
                        # print(" recv_x={}.".format(arr_recv), end='' )
                        print("                   [mse,rmse] recv_x={}.".format(arr_recv), end='' )
                        print(" orig_x={}.".format(arr_orig))
                    self.output_file(recv_in_x, recv_out_x_metric, orig_in_x, orig_out_x_metric)
                    # batch end
                update_rmse = meaners_RMSE['recv'].avg.numpy()
                if abs(update_rmse)>10:
                    print("    [Tower]["+keys_str+"] state exploid!")
                    stoper.status = True
                else:
                    stoper.update(meaners_RMSE['recv'].avg.numpy())
                for a in stages:
                    rd = [np.round(m.avg.numpy(),3) for m in [meaners_MSE[a], meaners_RMSE[a]]]
                    loss_rd[a] = rd
                rd = [np.round(meaner_mask.avg.numpy(),3), hist_epoches+Epoche, n_eps+Epoche]
                loss_rd['other'] = rd
                round_ep_1st = min([ep, n_epoch_1st-1])
                round_ep_2nd = max([ep-n_epoch_1st, -1])
                print("[Tower]["+keys_str+"] Epoche %d [iterate %d] round Auto[%d/%d] Cross[%d/%d] " % 
                      (Epoche, sh+1, round_ep_1st+1, n_epoch_1st, round_ep_2nd+1, n_epoch_2nd), end='')
                print("clip{%.3f,%.3f} \n           mean " % (state_clips[0],state_clips[1]), end='')
                print("<< Auto_1 = %.3f, Auto_2 = %.3f, Mask = %.3f >>  << Cross = %.3f >>" % (loss_rd['expd'][1], loss_rd['crss'][1], loss_rd['other'][0], loss_rd['rls'][1]))
                print("           mean [mse,rmse]  crss={}.".format(loss_rd['crss']), end='')
                print(" expd={}.".format(loss_rd['expd']), end='')
                print(" rls={}.".format(loss_rd['rls']))
                print("           mean [mse,rmse]  recv_x={}.".format(loss_rd['recv']), end='')
                print(" orig_x={}.".format(loss_rd['orig']))
                if not output_only:
                    if loss_rd['recv'][1] <= lowest_recv_rmse:
                        self.loss_recorder = loss_rd
                        self.save()
                        self.output_backup()
                        lowest_recv_rmse = loss_rd['recv'][1]
                        if enforce:
                            stoper.status = True
                    else:
                        print("           -- not saved -- min_recv_rmse="+str(np.round(lowest_recv_rmse, 4)))
                else:
                    self.loss_recorder = loss_rd
                    self.save()
                # epoch end
            # shuffle end
        if output_only:
            if bundle_tag:
                self.finished_tag = fixed_finished_tag
                self.save()
                print("    [Tower]["+keys_str+"] multi-keys Output Finished. ")
            else:
                print("    [Tower]["+keys_str+"] Output Only Finished. ")
            return
        self.finished_tag = True
        self.save()
        self.output_recover()
        run_finish = False # fully run or stopped
        if (sh==self.n_shuff-1) and (ep==n_epoch-1):
            if not stoper.status:
                run_finish = True
        if run_finish:
            print("    [Tower]["+keys_str+"] Finished, Epoche=%d. " % int(loss_rd['other'][1]))
        else:
            print("    [Tower]["+keys_str+"] Early Stopped, Epoche=%d. " % int(loss_rd['other'][1]))
        return










