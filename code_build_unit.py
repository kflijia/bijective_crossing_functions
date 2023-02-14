# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 11:36:55 2021

@author: Jia
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
import copy
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
#LeakyReLU = tf.keras.layers.LeakyReLU()
#from tensorflow.keras.regularizers import l2
#l2=tf.keras.regularizers.l2()
from os.path import exists
import math

import code_init_tower as tw
import code_load_data as dt
import code_bicross_extracter as bi
from code_utils import EarlyStop,AverageMeter

Dim_Bridge = 8
Batch_size = dt.batch_size
Dim_Repres = dt.representation_space_dim
verb = dt.verb

# ---------------
bce_loss_fn = tf.keras.losses.BinaryCrossentropy()
cs_loss_fn = tf.keras.losses.CosineSimilarity()

def binary_both_loss_fn(y_true, y_pred):
    bce = bce_loss_fn(y_true, y_pred)
    cs = cs_loss_fn(y_true, y_pred)
    if not hasattr(bce, "numpy"):
        bce = cs = tf.convert_to_tensor(float('nan'))
    return cs, bce

mse_loss_fn = tf.keras.losses.MeanSquaredError()
def rmse_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    mse = tf.reduce_mean(squared_difference, axis=1)
    rmse = tf.reduce_mean(tf.sqrt(mse), axis=0)
    return rmse

def both_loss_fn(y_true, y_pred): # support 3D
    mse = mse_loss_fn(y_true, y_pred)
    rmse = rmse_loss_fn(y_true, y_pred)
    if not hasattr(mse, "numpy"):
        mse = rmse = tf.convert_to_tensor(float('nan'))
    return mse, rmse

def nse_loss_fun(y_true, y_pred):
    NSE_numerator = tf.reduce_sum(tf.square(y_true-y_pred))
    NSE_denominator = tf.reduce_sum(tf.square(y_true-tf.reduce_mean(y_true, axis=0)))
    if NSE_denominator==0:
        return tf.convert_to_tensor(0)
    NSE = 1 - (NSE_numerator/NSE_denominator)
    if not hasattr(NSE, "numpy"):
        NSE = tf.convert_to_tensor(float('nan'))
    return NSE

from math import log2
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

def kl_divergence(p, q):
    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

# calculate the js divergence
def js_divergence(p, q):
	m = 0.5 * (p + q)
	return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def calculate_kld(p, q): # only support 2D
    if "strided_slice" in str(p[0]): # place holder!
        return float('nan')
    block = np.concatenate((p,q),0)
    _ = min_max_scaler.fit(block)
    p = min_max_scaler.transform(p)
    p = p.reshape([-1,])+1
    q = min_max_scaler.transform(q)
    q = q.reshape([-1,])+1
    return js_divergence(p,q)
#---------------------


#TODO brdg
class Bridge(layers.Layer):
    def __init__(self, n_steps, b_size=1, xy_gap=0, name="bridge", **kwargs):
        super(Bridge, self).__init__(name=name, **kwargs)
        self.n_steps = n_steps
        self.xy_gap = xy_gap
        self.dense_input = layers.Dense(1, kernel_initializer=initializers.RandomNormal(mean=0., stddev=0.1),
                                   bias_initializer=initializers.Zeros(), input_shape=(Dim_Repres,b_size), dtype='float32')
        self.gru_layer = layers.GRU(Dim_Bridge, stateful=False, dtype='float32')
        self.gru_bn = layers.BatchNormalization(dtype='float32', trainable=True) # True is for consistent behavior
        self.dense_relu = LeakyReLU(alpha=0.05)
        self.dense_bn = layers.BatchNormalization(dtype='float32', trainable=True)
        self.dense_output = layers.Dense(Dim_Repres, kernel_initializer=initializers.RandomNormal(mean=0., stddev=0.1), 
                                   bias_initializer=initializers.Zeros(), input_shape=(Batch_size,Dim_Bridge), dtype='float32')
        self.input_dim = Dim_Repres
        block_size = self.n_steps + self.xy_gap
        n_buff_case = block_size-1
        self.n_buff_case = n_buff_case
    def call(self, inputs_list, in_buffer):
        assert(isinstance(inputs_list, list))
        if isinstance(in_buffer, list) and in_buffer==[]:
            in_buffer = tf.convert_to_tensor(np.zeros([self.n_steps, Dim_Repres]),dtype='float32')
        in_len = inputs_list[0].shape[0]
        inputs = tf.constant([[[] for j in range(Dim_Repres)] for i in range(in_len)])
        for inputs_b in inputs_list:
            inputs_b = tf.expand_dims(inputs_b, -1)
            inputs = tf.concat([inputs,inputs_b], axis=2)
        inputs = self.dense_input(inputs) # 3D
        inputs = tf.reshape(inputs, [in_len,Dim_Repres]) # 2D
        #inputs_mat = tf.concat([self.buffer, inputs], axis=0)
        inputs_mat = tf.concat([in_buffer, inputs], axis=0)
        buffer_mat = inputs[-self.n_buff_case:,:]
        out_buffer = tf.identity(buffer_mat)
        x_set = tf.convert_to_tensor(np.zeros([0, self.n_steps, Dim_Repres]), dtype=inputs.dtype)
        for i in range(in_len):
            x_block = inputs_mat[i:(i+self.n_steps),:]
            x_set = tf.concat([x_set, tf.expand_dims(x_block,axis=0)], axis=0)
        gru_outputs = self.gru_bn(self.gru_layer(x_set))
        outputs = self.dense_output(self.dense_bn(self.dense_relu(gru_outputs)))
        outputs = tf.cast(outputs, dtype=inputs.dtype)
        outputs = [outputs] # b_size=1
        return outputs, out_buffer
    def get_weights(self, in_list):
        in_list.append(self.gru_layer.get_weights())
        in_list.append(self.gru_bn.get_weights())
        in_list.append(self.dense_relu.get_weights())
        in_list.append(self.dense_bn.get_weights())
        in_list.append(self.dense_output.get_weights())
        
        in_list.append(self.dense_input.get_weights())
        return in_list
    def set_weights(self, in_list):
        # TODO: Correct gru w, add input_w
        gru_w = in_list[0]
        if len(gru_w[2].shape)==1:
            replace_w = np.zeros([2,24])
            replace_w[0,:] = gru_w[2]
            in_list[0][2] = replace_w
        
        self.gru_layer.set_weights(in_list[0])
        
        self.gru_bn.set_weights(in_list[1])
        self.dense_relu.set_weights(in_list[2])
        self.dense_bn.set_weights(in_list[3])
        self.dense_output.set_weights(in_list[4])
        
        if len(in_list)>5:
            self.dense_input.set_weights(in_list[5])
        return

#TODO state
class Adapt_State(layers.Layer):
    def __init__(self, bundle_size=1, active_state=True, name="adapt_state", **kwargs):
        super(Adapt_State, self).__init__()
        layer_dense = layers.Dense(bundle_size, 
                                   kernel_initializer=initializers.Identity(), 
                                   bias_initializer=initializers.Zeros(), 
                                   input_shape=(Dim_Repres,bundle_size), dtype='float32', trainable=True)
        self.layer_dense = layer_dense
        self.outlet_tag = False
    def call(self, inputs, fake_buff=[]):
        if isinstance(inputs, list):
            in_len = inputs[0].shape[0]
            the_inputs = tf.constant([[[] for j in range(Dim_Repres)] for i in range(in_len)])
            for inputs_b in inputs:
                inputs_b = tf.expand_dims(inputs_b, -1)
                the_inputs = tf.concat([the_inputs,inputs_b], axis=2)
        else:
            the_inputs = inputs
        outputs = self.layer_dense(the_inputs)
        if self.outlet_tag:
            outputs_list = []
            bundle_size = outputs.shape[-1]
            for b in range(bundle_size):
                outputs_list.append(outputs[:,:,b])
            return outputs_list, fake_buff
        else:
            return outputs, fake_buff
    def get_weights(self, in_list):
        in_list.append(self.layer_dense.get_weights())
        return in_list
    def set_weights(self, in_list):
        self.layer_dense.set_weights(in_list[0])
        return



class strong_Adapt(layers.Layer):
    def __init__(self, n_layer=2, bundle_size=1, active_state=True, name="strong_adapt", **kwargs):
        super(strong_Adapt, self).__init__()
        if n_layer>4:
            n_layer = 4
        if n_layer<2:
            n_layer = 2
        self.layer_in = layers.Dense(bundle_size, 
                                   kernel_initializer=initializers.Identity(), 
                                   bias_initializer=initializers.Zeros(), 
                                   input_shape=(Dim_Repres,bundle_size), dtype='float32', trainable=True)
        self.dense_relu = LeakyReLU(alpha=0.05)
        self.dense_bn = layers.BatchNormalization(dtype='float32', trainable=True)
        self.states_flow = []
        for i in range(1, n_layer):
            layer_dense = layers.Dense(bundle_size, 
                                       kernel_initializer=initializers.Identity(), 
                                       bias_initializer=initializers.Zeros(), 
                                       input_shape=(Dim_Repres,bundle_size), dtype='float32', trainable=True)
            self.states_flow.append(layer_dense)
        self.outlet_tag = False
    def call(self, inputs, fake_buff=[]):
        if isinstance(inputs, list):
            in_len = inputs[0].shape[0]
            the_inputs = tf.constant([[[] for j in range(Dim_Repres)] for i in range(in_len)])
            for inputs_b in inputs:
                inputs_b = tf.expand_dims(inputs_b, -1)
                the_inputs = tf.concat([the_inputs,inputs_b], axis=2)
        else:
            the_inputs = inputs
        # --- call ---
        outputs = self.layer_in(the_inputs)
        outputs = self.dense_relu(outputs)
        outputs = self.dense_bn(outputs)
        for i in range(len(self.states_flow)):
            outputs = self.states_flow[i](outputs)
        if self.outlet_tag:
            outputs_list = []
            bundle_size = outputs.shape[-1]
            for b in range(bundle_size):
                outputs_list.append(outputs[:,:,b])
            return outputs_list, fake_buff
        else:
            return outputs, fake_buff
    def get_weights(self, in_list):
        in_list.append(self.layer_in.get_weights())
        in_list.append(self.dense_relu.get_weights())
        in_list.append(self.dense_bn.get_weights())
        for i in range(len(self.states_flow)):
            i_w = self.states_flow[i].get_weights()
            in_list.append(i_w)
        return in_list
    def set_weights(self, in_list):
        self.layer_in.set_weights(in_list[0])
        self.dense_relu.set_weights(in_list[1])
        self.dense_bn.set_weights(in_list[2])
        for i in range(len(self.states_flow)):
            self.states_flow[i].set_weights(in_list[i+3])
        return



#TODO unit
class Preserve_Unit(keras.Model):
    def __init__(self, the_out, the_in, in_data_obj, fname_set, name="preserve_unit", **kwargs):
        super(Preserve_Unit, self).__init__(name=name, **kwargs)        
        self.d_obj = in_data_obj
        if isinstance(the_out, str):
            self._name = "discv_unit"
            in_node_key = the_out # str
            in_causes_keys = the_in # list
        else: # Snode
            self._name = "train_unit"
            in_node_key = the_out.key
            self.tower_src = the_in.orig_tower
            in_causes_keys = [k for k in the_in.virtural_key] # list len=1
            # maybe key==virtural_key
            self.A_stack = the_in.call_state_flow
            self.vir_A_stack = the_in.call_virtural # no virtural_flow when key==virtural_key
            self.reset_vir_A = the_in.reset_virtural
            self.B_stack = the_out.call_state_flow
            if "refine" in fname_set[0]:
                in_node_key = the_out.virtural_key
                self.tower_rfn = the_out.orig_tower
                self.vir_B_stack = the_out.call_virtural # has virtural_flow
                self.reset_vir_B = the_out.reset_virtural
        # ---
        b_size_A = len(in_causes_keys)
        self.tower_A = tw.Tower(in_causes_keys, self.d_obj)
        self.tower_A.load()
        b_size_B = 1
        self.tower_B = tw.Tower(in_node_key, self.d_obj)
        self.tower_B.load()
        self.b_size_pair = [b_size_A, b_size_B]
        # ---
        self.run_fname = fname_set[0]
        self.done_fname = fname_set[1]
        output_fname = fname_set[2]
        self.output_fname_orig = output_fname+"_orig.csv"
        self.output_fname_recv = output_fname+"_recv.csv"
        self.keys_str = ''.join(in_causes_keys)+'_'+in_node_key
        # ----- training kit -----
        # - A_adapter -
        self.A_adapter = Adapt_State(b_size_A)
        self.A_adapter.outlet_tag = True
        # - bridge -
        step_num = self.d_obj.get_merged_step(in_causes_keys, in_node_key)
        assert step_num>0
        self.bridge = Bridge(n_steps=step_num, b_size=b_size_A)
        self.brg_buffer = []
        # - B_adapter -
        self.B_adapter = Adapt_State()
        self.B_adapter.outlet_tag = True
        # - A_top -
        self.A_top = tw.Tower_Top(self.tower_A.tw_base)
        self.A_top = self.tower_A.copy_top(self.A_top, "_A_top")
        # - B_top -
        self.B_top = tw.Tower_Top(self.tower_B.tw_base)
        self.B_top = self.tower_B.copy_top(self.B_top, "_B_top")
        if "train" in self._name:
            if not hasattr(the_in, "init_top"):
                the_in.init_top = the_in.get_top_flow()
            if not hasattr(the_out, "init_top"):
                the_out.init_top = the_out.get_top_flow()
            self.A_top.set_weights(the_in.init_top)
            self.B_top.set_weights(the_out.init_top)
        # --
        self.tower_A.tw_base.encoder.trainable = False
        self.tower_B.tw_base.encoder.trainable = False
        self.tower_A.tw_base.extracter.do_fun("disable")
        self.tower_B.tw_base.extracter.do_fun("disable")
        self.A_adapter.trainable = True
        self.A_top.decoder.trainable = True
        self.bridge.trainable = True
        self.B_adapter.trainable = True
        self.B_top.decoder.trainable = True
        # -- other --
        self.stages = ["crss", "expd", "mask", "recv", "orig"]
        recoder_A = {}
        recoder_B = {}
        recoder_brdg = {}
        for a in self.stages + ["NSE"]:
            recoder_A[a] = [float('nan'),float('nan')]
            recoder_B[a] = [float('nan'),float('nan')]
            recoder_brdg[a] = [float('nan'),float('nan')]
        recoder_brdg["state"] = [float('nan'), float('nan')]
        recoder_brdg["KLD"] = [float('nan'), float('nan')]
        # KLD repeated one value
        self.loss_recorder = {"A":recoder_A, "B":recoder_B, "brdg":recoder_brdg}
        self.prog_recorder = copy.deepcopy(self.loss_recorder)
        self.loss_recorder["n_ep_hist"] = 0
        self.loss_recorder["n_Ep"] = 0
        self.loss_recorder["hist_loss"] = []
        self.loss_recorder["hist_NSE"] = []
        self.loss_recorder["hist_B_NSE"] = []
        self.finished_tag = False
        # -- initialize --
        A_seeds = self.d_obj.get_seeds_bundle(self.tower_A.keys, Batch_size)
        B_seeds = self.d_obj.get_seeds_bundle(self.tower_B.keys, Batch_size)
        if "train" in self._name:
            self.A_x_dim = A_seeds[0][0].shape[-1]
            self.A_c_dim = A_seeds[1][0].shape[-1]
            src_seeds = self.d_obj.get_seeds_bundle(self.tower_src.keys, Batch_size)
            A_x = [self.convert_bundle(A_seeds[0]+src_seeds[0])] # maybe A_x==src_x
            A_c = [self.convert_bundle(A_seeds[1]+src_seeds[1])]
            A_seeds = [A_x, A_c]
            if "refine" in self.run_fname:
                self.B_x_dim = B_seeds[0][0].shape[-1]
                self.B_c_dim = B_seeds[1][0].shape[-1]
                rfn_seeds = self.d_obj.get_seeds_bundle(self.tower_rfn.keys, Batch_size)
                B_x = [self.convert_bundle(B_seeds[0]+rfn_seeds[0])] # maybe A_x==src_x
                B_c = [self.convert_bundle(B_seeds[1]+rfn_seeds[1])]
                B_seeds = [B_x, B_c]
        self.call(A_seeds, B_seeds)
        # calling seeds activates buff_list, and get reset in load()

        return
    def compare_loss(self, header="", metric='rmse', chng=False):
        idx = 1
        if metric=='mse':
            idx = 0
        loss_rd = copy.deepcopy(self.loss_recorder)
        prog_rd = copy.deepcopy(self.prog_recorder)
        hist_epoches = loss_rd['n_ep_hist']
        A_losses = loss_rd["A"]
        A_changes = prog_rd["A"]
        B_losses = loss_rd["B"]
        B_changes = prog_rd["B"]
        brdg_losses = loss_rd["brdg"]
        brdg_changes = prog_rd["brdg"]
        A_tw_losses = copy.deepcopy(self.tower_A.loss_recorder)
        B_tw_losses = copy.deepcopy(self.tower_B.loss_recorder)
        def print_loss_line(in_header, in_stages, in_losses, in_idx):
            out_line = in_header
            for a in in_stages:
                if a=="NSE":
                    in_idx = 0 # recv_NSE
                out_line += "{0: <6}  ".format(str(in_losses[a][in_idx])[:6])
            print(out_line)
            return
        def print_chng_line(in_header, in_stages, in_changes, in_idx):
            out_line = in_header
            for a in in_stages:
                if a=="NSE":
                    in_idx = 0 # recv_NSE
                a_sign = "+"
                a_value = in_changes[a][in_idx]
                if a_value<0:
                    a_sign = "-"
                    a_value = abs(a_value)
                out_line += "{0: <6}  ".format(a_sign+str(a_value)[:5])
            print(out_line)
            return
        # ------------
        summary_str = "performance: Epoches=%d, brdg recv=%.4f state=%.4f KLD=%.2f."
        print(header+summary_str % (hist_epoches, brdg_losses["recv"][idx], brdg_losses['state'][idx], brdg_losses['KLD'][idx]))
        print(header+"             [crss]  [expd]  [mask]  [recv]  [orig]  [NSE]  [state]  [KLD]")
        tw_stages = self.stages
        AB_stages = self.stages + ["NSE"]
        brdg_stages = self.stages + ["NSE", "state", "KLD"]
        # --- A_tw ---
        A_tw_losses["mask"]=[A_tw_losses['other'][0], A_tw_losses['other'][0]] # repeat
        print_loss_line(header+"A_tw losses: ", tw_stages, A_tw_losses, idx)
        # --- A ---
        if hist_epoches > 0:
            print_loss_line(header+"  A  losses: ", AB_stages, A_losses, idx)
            if chng:
                print_chng_line(header+"            ", AB_stages, A_changes, idx)
        # --- B_tw ---
        B_tw_losses["mask"]=[B_tw_losses['other'][0], B_tw_losses['other'][0]] # repeat
        print_loss_line(header+"B_tw losses: ", tw_stages, B_tw_losses, idx)
        # --- B ---
        if hist_epoches > 0:
            B_losses["state"] = B_losses["KLD"] = ['',''] # skip state and KLD
            print_loss_line(header+"  B  losses: ", AB_stages, B_losses, idx)
            if chng:
                B_changes["state"] = B_changes["KLD"] = ['',''] # skip state and KLD
                print_chng_line(header+"            ", AB_stages, B_changes, idx)
        # --- brdg ---
            print_loss_line(header+"brdg losses: ", brdg_stages, brdg_losses, idx)
            if chng:
                print_chng_line(header+"            ", brdg_stages, brdg_changes, idx)
        # --- loss_hist
            hist_loss_str = '->'.join([str(ls)[:5] for ls in loss_rd["hist_loss"]])
            print(header+"brdg recv history:", hist_loss_str)
            if "hist_NSE" in loss_rd:
                hist_NSE_str = '->'.join([str(ls)[:5] for ls in loss_rd["hist_NSE"]])
                print(header+"brdg NSE history:", hist_NSE_str)
            if "hist_B_NSE" in loss_rd:
                hist_B_NSE_str = '->'.join([str(ls)[:5] for ls in loss_rd["hist_B_NSE"]])
                print(header+"B NSE history:", hist_B_NSE_str)
        else:
            print("")
        # --- return min recv ---
        min_recv_rmse = np.mean([brdg_losses["recv"][1], B_losses["recv"][1]])
        if math.isnan(min_recv_rmse):
            min_recv_rmse = float('inf')
        return min_recv_rmse
    def convert_bundle(self, in_bundle):
        data_shape = list(in_bundle[0].shape[:-1])
        data_shape.extend([0])
        merged_data = tf.convert_to_tensor(np.zeros(data_shape), dtype=in_bundle[0].dtype)
        for the_data in in_bundle:
            merged_data = tf.concat([merged_data, the_data], axis=len(data_shape)-1)
        return merged_data
    def get_truth(self, inputs_A, inputs_B):
        A_truths = {}
        B_truths = {}
        A_x, A_c = inputs_A
        B_x, B_c = inputs_B
        if "train" in self._name:
            A_x, src_x = tf.split(A_x, [self.A_x_dim, -1], -1)
            A_c, src_c = tf.split(A_c, [self.A_c_dim, -1], -1)
            if "refine" in self.run_fname:
                B_x, rfn_x = tf.split(B_x, [self.B_x_dim, -1], -1)
                B_c, rfn_c = tf.split(B_c, [self.B_c_dim, -1], -1)
        # == A ==
        A_base_out = self.tower_A.call_base(A_x, A_c)
        A_truths['expd'], [A_truths['crss'], A_truths['text']], _, A_truths['mask'], A_truths['c_patch'], A_truths['recv'] = A_base_out
        # all bundle except A_truths['recv']
        A_x_orig = self.tower_A.descl_output(A_x) # bundle
        A_x_orig_merge = self.tower_A.tw_top.merge_output(A_x_orig)
        A_truths['orig'] = A_x_orig_merge # 2D tensor
        # == B ==
        B_base_out = self.tower_B.call_base(B_x, B_c)
        B_truths['expd'], [B_truths['crss'], B_truths['text']], _, B_truths['mask'], B_truths['c_patch'], B_truths['recv'] = B_base_out
        # all bundle except B_truths['recv']
        B_x_orig = self.tower_B.descl_output(B_x) # bundle
        B_x_orig_merge = self.tower_B.tw_top.merge_output(B_x_orig)
        B_truths['orig'] = B_x_orig_merge # 2D tensor
        return A_truths, B_truths
    def get_state(self, inputs_A, inputs_B):
        A_x, A_c = inputs_A
        B_x, B_c = inputs_B
        # ---
        if "train" in self._name:
            A_x, src_x = tf.split(A_x, [self.A_x_dim, -1], -1)
            A_c, src_c = tf.split(A_c, [self.A_c_dim, -1], -1)
            _, _, state_A_repr, _, _, _ = self.tower_src.call_base(src_x, src_c)
            state_A_repr = self.A_stack(state_A_repr)
            state_A_repr = self.vir_A_stack(state_A_repr)
            if "refine" in self.run_fname:
                B_x, rfn_x = tf.split(B_x, [self.B_x_dim, -1], -1)
                B_c, rfn_c = tf.split(B_c, [self.B_c_dim, -1], -1)
                _, _, state_B_repr, _, _, _ = self.tower_rfn.call_base(rfn_x, rfn_c)
                state_B_repr = self.B_stack(state_B_repr)
                state_B_repr = self.vir_B_stack(state_B_repr)
            else:
                _, _, state_B_repr, _, _, _ = self.tower_B.call_base(B_x, B_c)
                state_B_repr = self.B_stack(state_B_repr)
        else:
            _, _, state_A_repr, _, _, _ = self.tower_A.call_base(A_x, A_c)
            _, _, state_B_repr, _, _, _ = self.tower_B.call_base(B_x, B_c)
        # ---
        states = {}
        states['A_repr'] = state_A_repr # (bundle)
        states['B_repr'] = state_B_repr # (bundle)
        states['A_adpt'], _ = self.A_adapter(state_A_repr) # (bundle)
        states['B_adpt'], _ = self.B_adapter(state_B_repr) # (bundle)
        states['brdg'], self.brg_buffer = self.bridge(states['A_adpt'], self.brg_buffer) # (bundle)
        B_state = states['B_adpt'][0] # 2D, b_size = 1 
        brdg_state = states['brdg'][0] # 2D, b_size = 1 
        state_losses = both_loss_fn(B_state, brdg_state)
        state_KLD = calculate_kld(brdg_state, B_state)
        state_KLD = [state_KLD, state_KLD] # repeat
        return states, state_losses, state_KLD
    #TODO unit_call
    def call(self, inputs_A, inputs_B):
        A_truths, B_truths = self.get_truth(inputs_A, inputs_B)
        states, state_losses, state_KLD = self.get_state(inputs_A, inputs_B)
        A_preds = {}
        B_preds = {}
        brdg_preds = {}
        # ----
        # top_out = decd_output, rls_output, t_pair, shrk_merge_pair, mask_out_triple, mask_merge_pair
        # rls_output = rls_cipher, rls_avg, rls_text_mat, rls_cipher_metric
        # [omitted] shrk_out_pair = x_recv, x_recv_pred
        # shrk_merge_pair = x_merge, x_merge_pred 
        # mask_out_triple = mask_expd, mask_expd_bi, mask_recv_x_bi
        # [omitted] mask_merge_pair = mask_merge_x, mask_merge_x_bi
        #----
        A_decd_output, A_rls_output, A_shrk_output, A_shrk_merge_output, A_mask_output, _ = self.A_top(
            states['A_adpt'], A_truths['c_patch'], A_truths['expd'], A_truths['mask'])
        B_decd_output, B_rls_output, B_shrk_output, B_shrk_merge_output, B_mask_output, _ = self.B_top(
            states['B_adpt'], B_truths['c_patch'], B_truths['expd'], B_truths['mask'])
        brdg_decd_output, brdg_rls_output, brdg_shrk_output, brdg_shrk_merge_output, brdg_mask_output, _ = self.B_top(
            self.bridge(states['A_adpt'],self.brg_buffer)[0], B_truths['c_patch'], B_truths['expd'], B_truths['mask'])
        # ----
        A_preds['crss'] = A_decd_output # (bundle)
        A_preds['text'] = A_rls_output[-2] # rls_text_mat (bundle)
        A_preds['expd'] = A_rls_output[-1] # rls_cipher_metric (bundle)
        A_preds['recv'] = A_shrk_merge_output[1] # merged x_recv_pred, true mask applied (2D tensor)
        A_preds['mask'] = A_mask_output[0] # mask_expd, continuous value (bundle)
        A_orig_output = self.tower_A.descl_output(A_shrk_output[1])
        A_preds['orig'] = self.tower_A.tw_top.merge_output(A_orig_output) # (2D tensor)
        # ----
        B_preds['crss'] = B_decd_output
        B_preds['text'] = B_rls_output[-2] # rls_text_mat
        B_preds['expd'] = B_rls_output[-1] # rls_cipher_metric
        B_preds['recv'] = B_shrk_merge_output[1] # merged x_recv_pred, true mask applied
        B_preds['mask'] = B_mask_output[0] # mask_expd, continuous value
        B_orig_output = self.tower_B.descl_output(B_shrk_output[1])
        B_preds['orig'] = self.tower_B.tw_top.merge_output(B_orig_output)
        # ----
        brdg_preds['crss'] = brdg_decd_output
        brdg_preds['text'] = brdg_rls_output[-2] # rls_text_mat
        brdg_preds['expd'] = brdg_rls_output[-1] # rls_cipher_metric
        brdg_preds['recv'] = brdg_shrk_merge_output[1] # merged x_recv_pred, true mask applied
        brdg_preds['mask'] = brdg_mask_output[0] # mask_expd, continuous value
        brdg_orig_output = self.tower_B.descl_output(brdg_shrk_output[1])
        brdg_preds['orig'] = self.tower_B.tw_top.merge_output(brdg_orig_output)
        # ----
        return A_preds, B_preds, brdg_preds
        # 
    def get_loss(self, in_truths, in_preds):
        for a in ["crss", "expd", "text", "mask"]:
            in_truths[a] = self.convert_bundle(in_truths[a]) # 2D tensor
            in_preds[a] = self.convert_bundle(in_preds[a]) # 2D tensor
        crss_duo = both_loss_fn(in_truths['crss'], in_preds['crss']) # scalar
        #text_duo = both_loss_fn(in_truths['text'], in_preds['text'])
        expd_duo = both_loss_fn(in_truths['expd'], in_preds['expd'])
        mask_duo = binary_both_loss_fn(in_truths['mask'], in_preds['mask']) 
        out_losses = {} # loss_pair [mse, rmse]
        # -- for loss --
        out_losses['crss'] = crss_duo # scalar
        out_losses['expd'] = expd_duo
        out_losses['mask'] = mask_duo
        # -- for metric --
        out_losses['recv'] = both_loss_fn(in_truths['recv'], in_preds['recv'])
        out_losses['orig'] = both_loss_fn(in_truths['orig'], in_preds['orig'])
        NSE_recv = nse_loss_fun(in_truths['recv'], in_preds['recv'])
        NSE_orig = nse_loss_fun(in_truths['orig'], in_preds['orig'])
        out_losses['NSE'] = [NSE_recv, NSE_orig]
        return out_losses
    def output_reset(self):
        tw_colnames = self.d_obj.get_names(self.tower_B.keys)[0]
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
    def get_data_tuple(self):
        bundle_fn = self.d_obj.get_train_bundle
        A_x, A_c = bundle_fn(self.tower_A.keys)
        B_x, B_c = bundle_fn(self.tower_B.keys)
        if "train" in self._name:
            assert len(self.tower_A.keys)==1
            assert len(self.tower_src.keys)==1
            src_x, src_c = bundle_fn(self.tower_src.keys)
            A_x = [self.convert_bundle(A_x+src_x)] # maybe A_x==src_x
            A_c = [self.convert_bundle(A_c+src_c)]
            if "refine" in self.run_fname:
                assert len(self.tower_B.keys)==1
                assert len(self.tower_rfn.keys)==1
                rfn_x, rfn_c = bundle_fn(self.tower_rfn.keys)
                B_x = [self.convert_bundle(B_x+rfn_x)] # maybe B_x==rfn_x
                B_c = [self.convert_bundle(B_c+rfn_c)]
        dataset_tuple = tuple(A_x + A_c + B_x + B_c)
        return dataset_tuple
    def output_train_res(self):
        print("[Unit "+self.keys_str+"] output start... ")
        self.load()
        b_size_A = len(self.tower_A.keys)
        b_size_B = 1
        orig_x_dim = self.tower_B.tw_base.expander.config['orig_x_dim'][0]
        self.output_reset()
        dataset_tuple = self.get_data_tuple()
        dataset_train = tf.data.Dataset.from_tensor_slices(dataset_tuple)
        dataset_train = dataset_train.batch(Batch_size)
        init_stdout = sys.stdout
        full_recv_data = tf.convert_to_tensor(np.zeros([0,orig_x_dim*2]), dtype='float32')
        full_orig_data = tf.convert_to_tensor(np.zeros([0,orig_x_dim*2]), dtype='float32')
        for step, batch_train in enumerate(dataset_train):
            A_batch = batch_train[:int(b_size_A*2)]
            B_batch = batch_train[int(-b_size_B*2):]
            A_x_batch, A_c_batch = A_batch[:b_size_A], A_batch[-b_size_A:]
            B_x_batch, B_c_batch = B_batch[:b_size_B], B_batch[-b_size_B:]
            A_batches = [A_x_batch, A_c_batch]
            B_batches = [B_x_batch, B_c_batch]
            A_truths, B_truths = self.get_truth(A_batches, B_batches)
            brdg_truths = copy.deepcopy(B_truths)
            _, _, brdg_preds = self.call(A_batches, B_batches)
            # -- recv --
            recv_data = tf.concat([brdg_truths['recv'], brdg_preds['recv']], axis=1)
            full_recv_data = tf.concat([full_recv_data, recv_data], axis=0)
            recv_df = pd.DataFrame(recv_data.numpy())
            with open(self.output_fname_recv, 'a+') as f_recv:
                sys.stdout = f_recv
                print(recv_df.to_string(index=False, header=False))
            # -- orig --
            orig_data = tf.concat([brdg_truths['orig'], brdg_preds['orig']], axis=1)
            full_orig_data = tf.concat([full_orig_data, orig_data], axis=0)
            orig_df = pd.DataFrame(orig_data.numpy())
            with open(self.output_fname_orig, 'a+') as f_orig:
                sys.stdout = f_orig
                print(orig_df.to_string(index=False, header=False))
            sys.stdout = init_stdout
        NSE_recv = nse_loss_fun(full_recv_data[:,:orig_x_dim], full_recv_data[:,orig_x_dim:]).numpy()
        NSE_orig = nse_loss_fun(full_orig_data[:,:orig_x_dim], full_orig_data[:,orig_x_dim:]).numpy()
        print("[Unit "+self.keys_str+"] output finished, NSE_recv =", str(NSE_recv)[:5],", NSE_orig =", str(NSE_orig)[:5])
        return
    def output_performance(self):
        if self.finished_tag:
            print("[Unit "+self.keys_str+"] training finished already.")
        else:
            print("[Unit "+self.keys_str+"] is in training...")
        print("[Unit "+self.keys_str+"] Performance:")
        _ = self.compare_loss()
        return
    #TODO unit_train
    def do_train(self, n_unit_shuff=2, n_unit_epoch=3):
        keys_str = self.keys_str
        self.stopper_toler = self.tower_B.stopper_toler
        if self.finished_tag:
            print("[Unit "+keys_str+"] training finished already.")
            print("[Unit "+keys_str+"] History:")
            min_global_recv_rmse = self.compare_loss()
            return
        # === prepare ===
        print("[Unit "+keys_str+"] History:")
        min_global_recv_rmse = self.compare_loss()
        meaners_A = {}
        meaners_B = {}
        meaners_brdg = {}
        for a in self.stages+["NSE"]: # NSE: recv + orig
            meaners_A[a] = [AverageMeter(), AverageMeter()]
            meaners_B[a] = [AverageMeter(), AverageMeter()]
            meaners_brdg[a] = [AverageMeter(), AverageMeter()]
        meaners_brdg["state"] = [AverageMeter(), AverageMeter()]
        meaners_brdg["KLD"] = [AverageMeter(), AverageMeter()] # repeat KLD
        meaners_dict = {"A":meaners_A, "B":meaners_B, "brdg":meaners_brdg}
        stoper = EarlyStop(patience=4, tolerance=self.stopper_toler)
        optimizer_out = Adam(learning_rate=1e-3, clipnorm=1.0, clipvalue=0.5)
        optimizer_mask = Adam(learning_rate=1e-3, clipnorm=1.0, clipvalue=0.5)
        #optimizer_state = Adam(learning_rate=1e-3, clipnorm=1.0, clipvalue=0.5)
        state_clips = [float('inf'), -float('inf')] # [min, max]
        self.loss_recorder["n_Ep"] = 0
        # === create models ===
        b_size_A, b_size_B = self.b_size_pair
        # - A -
        input_1 = layers.Input(shape=(Dim_Repres,b_size_A), batch_size=Batch_size, dtype='float32')
        input_1_msk = [layers.Input(shape=(Dim_Repres), batch_size=Batch_size, dtype='float32') for b in range(b_size_A)]
        adapted_A, _ = self.A_adapter(input_1)
        outputs_auto_A = self.A_top.decoder(adapted_A)
        outputs_mask_A = self.A_top.masker(input_1_msk)
        self.model_tw_A = keras.Model(input_1, outputs_auto_A)
        self.model_mask_A = keras.Model(input_1_msk, outputs_mask_A)
        # - brdg -
        input_2 = layers.Input(shape=(Dim_Repres,b_size_A), batch_size=Batch_size, dtype='float32')
        input_2_st = layers.Input(shape=(Dim_Repres,b_size_A), batch_size=Batch_size, dtype='float32')
        input_2_msk = [layers.Input(shape=(Dim_Repres), batch_size=Batch_size, dtype='float32')]
        brdg_state, _ = self.bridge(self.A_adapter(input_2_st)[0], []) # [1] fake buff
        self.model_state = keras.Model(input_2_st, brdg_state)
        outputs_auto_brdg = self.B_top.decoder(self.bridge(self.A_adapter(input_2)[0], [])[0])
        self.model_tw_brdg = keras.Model(input_2, outputs_auto_brdg)
        outputs_mask_brdg = self.B_top.masker(input_2_msk)
        self.model_mask_brdg = keras.Model(input_2_msk, outputs_mask_brdg)
        # - B -
        input_3 = layers.Input(shape=(Dim_Repres,b_size_B), batch_size=Batch_size, dtype='float32')
        input_3_msk = [layers.Input(shape=(Dim_Repres), batch_size=Batch_size, dtype='float32')]
        adapted_B, _ = self.B_adapter(input_3)
        outputs_auto_B = self.B_top.decoder(adapted_B)
        outputs_mask_B = self.B_top.masker(input_3_msk)
        self.model_tw_B = keras.Model(input_3, outputs_auto_B)
        self.model_mask_B = keras.Model(input_3_msk, outputs_mask_B)
        # - compile -
        self.model_tw_A.compile(optimizer=optimizer_out)
        self.model_state.compile(optimizer=optimizer_out)
        self.model_tw_brdg.compile(optimizer=optimizer_out)
        self.model_tw_B.compile(optimizer=optimizer_out)
        self.model_mask_A.compile(optimizer=optimizer_mask)
        self.model_mask_brdg.compile(optimizer=optimizer_mask)
        self.model_mask_B.compile(optimizer=optimizer_mask)
        # === data ===
        b_size_A = len(self.tower_A.keys)
        b_size_B = 1
        dataset_tuple = self.get_data_tuple()
        # === start ===
        print("******** "+keys_str+" Unit no-shuffle training batch_size=%d ********" % (Batch_size))
        for sh in range(n_unit_shuff):
            if stoper.status:
                print("    ["+keys_str+" Unit] Stopped ahead of shuff="+str(sh+1)+".")
                sh -= 1
                break
            for ep in range(n_unit_epoch):
                if stoper.status:
                    print("    ["+keys_str+" Unit] Stopped ahead of shuff="+str(sh+1)+" epoche="+str(ep+1)+".")
                    ep -= 1
                    break
                for MODE in ["Mode AB", "Mode brdg", "Mode Wrap"]:
                    if MODE=="Mode Wrap" and not stoper.status: # only Wrap in the last epoch
                        break
                    print("    ["+keys_str+" Unit] [iterate %d][epoche %d]" % (sh+1, ep+1), end='')
                    print(" <"+MODE+">")
                    # -- meaner --
                    for obj in ["A","B","brdg"]:
                        obj_stages = self.stages+["NSE"]
                        if obj=="brdg":
                            obj_stages = self.stages+["NSE", "state", "KLD"]
                        for a in obj_stages:
                            for idx in [0,1]:
                                meaners_dict[obj][a][idx].reset()
                    # -- dataset --
                    dataset_train = tf.data.Dataset.from_tensor_slices(dataset_tuple)
                    dataset_train = dataset_train.batch(Batch_size)
                    for step, batch_train in enumerate(dataset_train):
                        A_batch = batch_train[:int(b_size_A*2)]
                        B_batch = batch_train[int(-b_size_B*2):]
                        A_x_batch, A_c_batch = A_batch[:b_size_A], A_batch[-b_size_A:]
                        B_x_batch, B_c_batch = B_batch[:b_size_B], B_batch[-b_size_B:]
                        A_batches = [A_x_batch, A_c_batch]
                        B_batches = [B_x_batch, B_c_batch]
                        A_truths, B_truths = self.get_truth(A_batches, B_batches)
                        brdg_truths = copy.deepcopy(B_truths)
                        if MODE=="Mode AB" or MODE=="Mode Wrap":
                            with tf.GradientTape(persistent=True) as tape:
                                A_preds, B_preds, brdg_preds = self.call(A_batches, B_batches)
                                A_losses = self.get_loss(A_truths, A_preds)
                                B_losses = self.get_loss(B_truths, B_preds)
                                loss_A_1 = A_losses['expd'][1]  # loss setting
                                loss_A_2 = A_losses['crss'][1]
                                loss_A_mask = A_losses['mask'][1]
                                loss_B_1 = B_losses['expd'][1]
                                loss_B_2 = B_losses['crss'][1]
                                loss_B_mask = B_losses['mask'][1]  # loss setting, scalar
                            grad_A_1 = tape.gradient(loss_A_1, self.model_tw_A.trainable_weights)
                            _ = optimizer_out.apply_gradients(zip(grad_A_1, self.model_tw_A.trainable_weights))
                            grad_A_2 = tape.gradient(loss_A_2, self.model_tw_A.trainable_weights)
                            _ = optimizer_out.apply_gradients(zip(grad_A_2, self.model_tw_A.trainable_weights))
                            grad_A_mask = tape.gradient(loss_A_mask, self.model_mask_A.trainable_weights)
                            _ = optimizer_mask.apply_gradients(zip(grad_A_mask, self.model_mask_A.trainable_weights))
                            grad_B_1 = tape.gradient(loss_B_1, self.model_tw_B.trainable_weights)
                            _ = optimizer_out.apply_gradients(zip(grad_B_1, self.model_tw_B.trainable_weights))
                            grad_B_2 = tape.gradient(loss_B_2, self.model_tw_B.trainable_weights)
                            _ = optimizer_out.apply_gradients(zip(grad_B_2, self.model_tw_B.trainable_weights))
                            grad_B_mask = tape.gradient(loss_B_mask, self.model_mask_B.trainable_weights)
                            _ = optimizer_mask.apply_gradients(zip(grad_B_mask, self.model_mask_B.trainable_weights))
                            brdg_losses = self.get_loss(brdg_truths, brdg_preds)
                            # - state loss - 
                            states, brdg_losses['state'], brdg_losses['KLD'] = self.get_state(A_batches, B_batches)
                        else: # Mode brdg
                            with tf.GradientTape(persistent=True) as tape:
                                A_preds, B_preds, brdg_preds = self.call(A_batches, B_batches)
                                brdg_losses = self.get_loss(brdg_truths, brdg_preds)
                                loss_brdg_1 = brdg_losses['expd'][1]  # loss setting, scalar
                                #loss_brdg_2 = brdg_losses['crss'][1]  # loss setting, scalar
                                loss_brdg_mask = brdg_losses['mask'][1]  # loss setting, scalar
                                # - state loss - 
                                states, brdg_losses['state'], brdg_losses['KLD'] = self.get_state(A_batches, B_batches)
                                loss_brdg_state = brdg_losses['state'][1] #rmse
                            grad_brdg_1 = tape.gradient(loss_brdg_1, self.model_tw_brdg.trainable_weights)
                            _ = optimizer_out.apply_gradients(zip(grad_brdg_1, self.model_tw_brdg.trainable_weights))
                            #grad_brdg_2 = tape.gradient(loss_brdg_2, self.model_tw_brdg.trainable_weights)
                            # _ = optimizer_out.apply_gradients(zip(grad_brdg_2, self.model_tw_brdg.trainable_weights))
                            grad_brdg_mask = tape.gradient(loss_brdg_mask, self.model_mask_brdg.trainable_weights)
                            _ = optimizer_mask.apply_gradients(zip(grad_brdg_mask, self.model_mask_brdg.trainable_weights))
                            grad_brdg_st = tape.gradient(loss_brdg_state, self.model_state.trainable_weights)
                            _ = optimizer_out.apply_gradients(zip(grad_brdg_st, self.model_state.trainable_weights))
                            A_losses = self.get_loss(A_truths, A_preds)
                            B_losses = self.get_loss(B_truths, B_preds)
                        state_brdg = states['brdg'][0] # 2D tensor
                        state_min = state_max = state_mean = float('nan')
                        if hasattr(state_brdg, "numpy"):
                            state_min = tf.math.reduce_min(state_brdg)
                            state_max = tf.math.reduce_max(state_brdg)
                            state_mean = tf.math.reduce_mean(state_brdg)
                            state_clips[0] = min(state_clips[0], state_min.numpy())
                            state_clips[1] = max(state_clips[1], state_max.numpy())
                        # --- update meaners ---
                        losses_dict = {"A":A_losses, "B":B_losses, "brdg":brdg_losses}
                        for obj in ["A","B","brdg"]:
                            obj_stages = self.stages+["NSE"]
                            if obj=="brdg":
                                obj_stages = self.stages+["NSE", "state", "KLD"]
                            for a in obj_stages:
                                for idx in [0,1]:
                                    updt_loss = losses_dict[obj][a][idx]
                                    if hasattr(updt_loss, "numpy"):
                                        updt_loss = updt_loss.numpy()
                                    meaners_dict[obj][a][idx].update(updt_loss)
                        # --- log display ---
                        if (step % 100 == 0 and verb >= 2) or (step % 50 == 0 and verb >= 3):
                            print("        ["+keys_str+"]["+MODE+"].... batches "+str(step)+": clip{%.3f,%.3f}%.1f" % 
                                  (state_min, state_max, state_mean),end='')
                            print("...[brdg state] rmse = {0: <5}  ".format(str(losses_dict["brdg"]['state'][1].numpy())[:5]), end='')
                            print("KLD = {0: <5} ".format(str(losses_dict["brdg"]["KLD"][1])[:5]))
                            for obj in ["A","B","brdg"]:
                                print("             ["+obj+" rmse]", end='')
                                for a in self.stages:
                                    print(" "+a+"={0: <5},".format(str(losses_dict[obj][a][1].numpy())[:5]), end='' )
                                print(" ["+obj+" NSE]", end='')
                                print(" NSE_recv={0: <5}.".format(str(losses_dict[obj]['NSE'][0].numpy())[:5]), end='')
                                print(" NSE_orig={0: <5}.".format(str(losses_dict[obj]['NSE'][1].numpy())[:5]))
                        # batch end
                    # --- update loss_recorder ---
                    for obj in ["A","B","brdg"]:
                        obj_stages = self.stages+["NSE"]
                        if obj=="brdg":
                            obj_stages = self.stages+["NSE", "state", "KLD"]
                        for a in obj_stages:
                            for idx in [0, 1]:
                                self.loss_recorder[obj][a][idx] = meaners_dict[obj][a][idx].avg
                                self.prog_recorder[obj][a][idx] = meaners_dict[obj][a][idx].chng
                    self.loss_recorder["n_ep_hist"] += 1
                    self.loss_recorder["n_Ep"] += 1
                    self.loss_recorder["hist_loss"].append(meaners_dict["brdg"]["recv"][1].avg)
                    self.loss_recorder["hist_NSE"].append(meaners_dict["brdg"]["NSE"][0].avg)
                    self.loss_recorder["hist_B_NSE"].append(meaners_dict["B"]["NSE"][0].avg)
                    new_global_recv_rmse = self.compare_loss(header="    ["+keys_str+"]["+MODE+"]")
                    stoper.update(new_global_recv_rmse)
                    self.save_running()
                    if new_global_recv_rmse < min_global_recv_rmse:
                        min_global_recv_rmse = new_global_recv_rmse
                        self.save_best()
                    else:
                        global_rmse_brdg_str = str(np.round(meaners_dict["brdg"]["recv"][1].avg, 4))
                        global_rmse_B_str = str(np.round(meaners_dict["B"]["recv"][1].avg, 4))
                        print("     -- no progress -- min_global_recv_rmse="+str(np.round(min_global_recv_rmse, 4)),
                              "(brdg="+global_rmse_brdg_str+", B="+global_rmse_B_str+")")
                    # MODE end
                # epoch end
            # iterate end
        self.save_done()
        run_finish = False # whether fully run
        if (sh==n_unit_shuff-1) and (ep==n_unit_epoch-1):
            run_finish = True
        if run_finish:
            print("    ["+keys_str+" Unit] Finished, Total Epoche=%d. " % (self.loss_recorder["n_ep_hist"]))
        else:
            print("    ["+keys_str+" Unit] Early Stopped, Total Epoche=%d. " % (self.loss_recorder["n_ep_hist"]))
        return self.finished_tag
    def reset(self):
        self.finished_tag = False
        os.remove(self.done_fname)
        return
    def save_running(self):
        weights_kit = {}
        weights_kit["A_adapt"] = self.A_adapter.get_weights([])
        weights_kit["A_top"] = self.A_top.get_weights([]) # including cross_weights
        weights_kit["bridge"] = self.bridge.get_weights([])
        weights_kit["B_adapt"] = self.B_adapter.get_weights([])
        weights_kit["B_top"] = self.B_top.get_weights([])
        saved_list = [weights_kit, self.loss_recorder, self.prog_recorder, self.finished_tag]
        dt.save_obj(saved_list, self.run_fname)
        return
    def save_best(self):
        run_best_fname = self.run_fname+"_best"
        mv_command = "mv "+self.run_fname+" "+run_best_fname
        os.system(mv_command)
        self.save_running()
        return
    def save_done(self):
        #self.load()
        self.finished_tag = True
        self.save_running()
        ln_command = "ln -s ../"+self.run_fname+" "+self.done_fname
        run_best_fname = self.run_fname+"_best"
        if exists(run_best_fname):
            best_saved = dt.load_obj(run_best_fname)
            best_saved[3] = True # finished_tag
            dt.save_obj(best_saved, run_best_fname)
            ln_command = "ln -s ../"+run_best_fname+" "+self.done_fname
        if not exists(self.done_fname):
            os.system(ln_command)
        return
    def reset_running(self):
        reset_fnames = []
        if exists(self.run_fname):
            saved_list = dt.load_obj(self.run_fname)
            saved_list[3] = False # finished_tag
            dt.save_obj(saved_list, self.run_fname)
            reset_fnames.append("_Run")
        else:
            self.finished_tag=False
            self.save_running()
            reset_fnames.append("_Run")
        run_best_fname = self.run_fname+"_best"
        if exists(run_best_fname):
            best_saved = dt.load_obj(run_best_fname)
            best_saved[3] = False # finished_tag
            dt.save_obj(best_saved, run_best_fname)
            reset_fnames.append("_Best")
        os.system("rm -f "+self.done_fname)
        reset_fnames.append("_Done")
        print("    ["+self.keys_str+" Unit] reset fnames: ",reset_fnames)
        return
    def load(self):
        keys_str = self.keys_str
        if "train" in self._name:
            type_logger = "Unit_Train"
            self.reset_vir_A()
            if "refine" in self.run_fname:
                type_logger = "Unit_Refine"
                self.reset_vir_B()
            load_tag = self.do_load()
            if load_tag:
                print("["+type_logger+" "+keys_str+"] successfully loaded.")
            else:
                discv_fname_set = dt.train2discv_unit_fnames(self.run_fname)
                discv_run_fname = discv_fname_set[0]
                if exists(discv_run_fname+"_best"):
                    discv_run_fname = discv_run_fname+"_best"
                if exists(discv_run_fname):
                    discv_saved_list = dt.load_obj(discv_run_fname)
                    discv_weights_kit = discv_saved_list[0]
                    self.bridge.set_weights(discv_weights_kit["bridge"]) # only borrow bridge!!!
                    print("["+type_logger+" "+keys_str+"] discv_unit bridge weights are initially loaded.")
                else:
                    print("["+type_logger+" "+keys_str+"] no discv_unit weights! bridge randomly initialized.")
        else:
            _ = self.do_load()
        return
    def do_load(self):
        run_best_fname = self.run_fname+"_best"
        if exists(run_best_fname): # run_fname also exists
            saved_list = dt.load_obj(run_best_fname)
        elif exists(self.run_fname):
            saved_list = dt.load_obj(self.run_fname)
        if "saved_list" in locals():
            weights_kit, saved_loss_recorder, saved_prog_recorder, self.finished_tag = saved_list
            self.A_adapter.set_weights(weights_kit["A_adapt"])
            self.A_top.set_weights(weights_kit["A_top"]) # setting cross_weights, in case changed
            self.bridge.set_weights(weights_kit["bridge"])
            self.B_adapter.set_weights(weights_kit["B_adapt"])
            self.B_top.set_weights(weights_kit["B_top"])
            for obj in saved_loss_recorder:
                if not isinstance(saved_loss_recorder[obj], dict):
                    self.loss_recorder[obj] = saved_loss_recorder[obj]
                else:
                    for a in saved_loss_recorder[obj]:
                        self.loss_recorder[obj][a] = saved_loss_recorder[obj][a]
            for obj in saved_prog_recorder:
                for a in saved_prog_recorder[obj]:
                    self.prog_recorder[obj][a] = saved_prog_recorder[obj][a]
            return True
        else:
            return False




