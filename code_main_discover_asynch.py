# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 02:08:47 2022

@author: Jia
"""
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.1

import numpy as np

tf.compat.v1.enable_eager_execution()
#tf.enable_eager_execution()
import code_init_tower as tw
import code_build_unit as un
import code_load_data as dt
#import pandas as pd
import sys

action_str = sys.argv[1]
identity_str = sys.argv[2]

other_str=''
if len(sys.argv)>3:
    other_str = sys.argv[3]

#action_str = "tower"
#identity_str = "ghi"

run_mode = ''
if 'tower' in action_str:
    run_mode = 'tw'
    tw_keys = identity_str
else:
    # unit
    run_mode = 'un'
    causes_keys_str, node_key = identity_str.split('_')
    causes_keys = [char for char in causes_keys_str]
    if 'backwd' in action_str:
        bk_tag = True
    else:
        bk_tag = False
    if not any(a in action_str for a in ['single','backwd','forwd']):
        raise RuntimeError("[Main_Asynch] wrong action: "+action_str)

#---
data_obj = dt.Data_Store()
data_obj.load()
# run tower
if run_mode=='tw':
    key_tags = [k in data_obj.node_keys for k in tw_keys]
    if not all(key_tags):
        raise RuntimeError("[Main_Discover_Asynch] wrong tower node key.")
    print("[Main_Discover_Asynch] run tower keys = ", tw_keys)
    tower = tw.Tower(tw_keys, data_obj)
    tower.load()
    
    if other_str=="enforce":
        tower.finished_tag = False
        tower.do_train(n_shuff=8, n_epoch_rounds=[3,0], enforce=True)
    elif other_str=="output":
        tower.do_train(n_shuff=8, n_epoch_rounds=[3,2], output_only=True)
    else:
        tower.do_train(n_shuff=8, n_epoch_rounds=[3,2])
    exit()

# run_mode=='un'
if bk_tag:
    log_str = 'Backprob '+action_str
else:
    log_str = 'Forward '+action_str

if len(node_key)>1 or not node_key in data_obj.node_keys:
    raise RuntimeError("[Main_Discover_Asynch] "+log_str+" task: wrong node key "+node_key)
    exit()

# run unit
fname_set = dt.get_discv_unit_fnames(node_key, causes_keys, bk_tag)
new_unit = un.Preserve_Unit(node_key, causes_keys, data_obj, fname_set)
new_unit.load()

if other_str=="enforce":
    new_unit.reset_done()
    print("[Main_Discover_Asynch] "+log_str+" "+causes_keys_str+"=>"+node_key+" training...")
    _ = new_unit.do_train(n_unit_shuff=3, n_unit_epoch=3)
    new_unit.output_train_res()
elif other_str=="output":
    new_unit.output_train_res()
elif other_str=="perform":
    new_unit.output_performance()
else:
    print("[Main_Discover_Asynch] "+log_str+" "+causes_keys_str+"=>"+node_key+" training...")
    _ = new_unit.do_train(n_unit_shuff=2, n_unit_epoch=3)
    new_unit.output_train_res()

print("[Main_Discover_Asynch] "+log_str+" "+causes_keys_str+"=>"+node_key+" task done.")






