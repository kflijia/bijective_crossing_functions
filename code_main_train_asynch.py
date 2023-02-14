# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:40:02 2022

@author: Jia
"""
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

config = tf.compat.v1.ConfigProto#tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.1

import numpy as np

tf.compat.v1.enable_eager_execution()
#tf.enable_eager_execution()
import code_load_data as dt
import code_build_unit as un
import code_train_graph as gp
#import pandas as pd
import sys

predr_id = sys.argv[1]
identity_str = sys.argv[2]
other_str=''
if len(sys.argv)>3:
    other_str = sys.argv[3]


if "predr" in predr_id:
    train_type = "train"
else:
    train_type = "refine"

train_rout = identity_str.split('_')
source_key = train_rout[0]
cause_key, node_key = train_rout[-2], train_rout[-1]
print("[Main_Train_Asynch] "+predr_id+" training "+"=>".join(train_rout)+"...")

# ---
data_obj = dt.Data_Store()
data_obj.load()

if len(node_key)>1 or not node_key in data_obj.node_keys:
    raise RuntimeError("[Main_Train_Asynch] task "+identity_str+": wrong node key "+node_key)
if len(node_key)>1 or not cause_key in data_obj.node_keys:
    raise RuntimeError("[Main_Train_Asynch] task "+identity_str+": wrong cause key "+cause_key)

def stack_node_vir(vir_stack, vir_rout, top_cause):
    for i in range(1, len(vir_rout)): # until the outlet
        n = vir_rout[i]
        n_cause = vir_rout[i-1]
        rt_node = gp.Status_Node(n, data_obj, predr_id)
        rt_node.load()
        if i==len(vir_rout)-1: # the outlet
            cause_in_list, cause_state_list = rt_node.get_cause_flow(n_cause, top_cause)
        else:
            cause_in_list, cause_state_list = rt_node.get_cause_flow(n_cause)
        bridge_n_step = cause_in_list[2]
        vir_stack.append(-1)
        vir_stack.append(cause_in_list[0])
        vir_stack.append(bridge_n_step)
        vir_stack.append(cause_in_list[1])
        st_len = len(cause_state_list)
        for st in range(st_len):
            st_weights = cause_state_list[st]
            if st == (st_len-1):
                vir_stack.append(-1)
            else:
                vir_stack.append(0)
            vir_stack.append(st_weights)
    outlet_init_top = rt_node.get_top_flow(top_cause)
    return vir_stack, outlet_init_top
# ---
# Node B
the_node = gp.Status_Node(node_key, data_obj, predr_id)
the_node.load()
if train_type=="train":
    the_node.buildup("base")
    the_node.init_top = the_node.get_top_flow(train_rout[-2]) # may be orig_top
else:
    if the_node.n_flows==0:
        the_node.buildup("base")
        the_node.init_top = the_node.get_top_flow(train_rout[-2]) # must be orig_top
    else:
        refine_rout = []
        refine_source_key = node_key
        while len(the_node.flows_causes)>0:
            refine_rout.append(refine_source_key)
            refine_source_key = the_node.flows_causes[0]
            the_node = gp.Status_Node(refine_source_key, data_obj, predr_id)
            the_node.load()
        refine_rout.append(refine_source_key)    
        refine_rout.reverse()
        print("[Main_Train_Asynch] "+predr_id+" refining B_source "+"=>".join(refine_rout)+".")
        the_node = gp.Status_Node(refine_source_key, data_obj, predr_id)
        the_node.virtural_key = node_key
        the_node.load()
        the_node.virtural_stack, the_node.init_top = stack_node_vir([], refine_rout, train_rout[-2])
        # init_top must not be orig_top
        the_node.buildup("base")
        the_node.buildup("virtural")
        
# Node A
source_node = gp.Status_Node(source_key, data_obj, predr_id) # with virtural stack
source_node.virtural_key = cause_key
source_node.load()
source_node.virtural_stack, source_node.init_top = stack_node_vir([], train_rout[:-1], train_rout[-3])
# init_top must not orig_top
source_node.buildup("virtural")
# for both A and B, no need to buildup(cause)

cause_fname_set = dt.get_train_unit_fnames(node_key, cause_key, predr_id)
new_unit = un.Preserve_Unit(the_node, source_node, data_obj, cause_fname_set)
new_unit.load()

if other_str=="enforce":
    new_unit.reset_done()
    _ = new_unit.do_train(n_unit_shuff=3, n_unit_epoch=3)
    new_unit.output_train_res()
elif other_str=="output":
    new_unit.output_train_res()
else:
    #print("[Main_Train_Asynch] "+predr_id+" training "+"=>".join(train_rout)+"...")
    _ = new_unit.do_train(n_unit_shuff=2, n_unit_epoch=3)
    new_unit.output_train_res()

print("[Main_Train_Asynch] "+predr_id+" "+"=>".join(train_rout)+" output done.")




