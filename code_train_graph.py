# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:02:41 2021

@author: Jia
"""

import numpy as np
import tensorflow as tf
import copy
from datetime import datetime
from time import sleep
from tensorflow import keras
import sys
import itertools
from os.path import exists
import os
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import code_init_tower as tw
import code_build_unit as un
import code_load_data as dt
from code_utils import AverageMeter
from code_display_graph import Show_Box
from code_display_graph import Show_general

Batch_size = dt.batch_size
Dim_Repres = dt.representation_space_dim

verb = dt.verb

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
    mse = tf.reduce_mean(squared_difference, axis=-1)
    rmse = tf.reduce_mean(tf.sqrt(mse), axis=-1)
    #mse = (np.square(y_true - y_pred)).mean(axis=1)
    #rmse = np.mean(np.sqrt(mse))
    return rmse

def both_loss_fn(y_true, y_pred):
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

#--------------------------------

class Unit_Manager(object):
    def __init__(self, in_graph_obj, in_target_key):
        super(Unit_Manager, self).__init__()
        self.g_obj = in_graph_obj
        self.target_key = in_target_key
        keys_queue = []
        keys_queue.append(self.target_key)
        keys_done = []
        full_set = []
        single_set = []
        multi_set = []
        while len(keys_queue)>0:
            k = keys_queue.pop()
            if not k in keys_done:
                k_causes = self.g_obj.nodes_in(k)
                if len(k_causes)>0:
                    full_set.append([''.join(k_causes), k])
                    for c in k_causes:
                        keys_queue.insert(0, c)
                        if not [c,k] in full_set: # equal to len(k_causes)=1
                            single_set.append([c,k])
                    keys_done.append(k)
                    for L in range(2,len(k_causes)):
                        for k_causes_comb in itertools.combinations(k_causes, L):
                            multi_set.append([''.join(k_causes_comb), k])
        self.full_set = full_set
        self.single_set = single_set
        self.multi_set = multi_set
        self.full_evals = {}
        self.single_evals = {}
        self.multi_evals = {}
        self.task_demands = []
        self.tower_demands = []
        return
    def get_weights(self, in_container):
        in_container.extend([[],[],[]])
        in_container[0] = [self.full_set, self.single_set, self.multi_set]
        in_container[1] = [self.full_evals, self.single_evals, self.multi_evals]
        in_container[2] = [self.task_demands, self.tower_demands]
        return in_container
    def set_weights(self, in_weights):
        self.full_set, self.single_set, self.multi_set = in_weights[0]
        self.full_evals, self.single_evals, self.multi_evals = in_weights[1]
        self.task_demands, self.tower_demands = in_weights[2]
        return
    def insert_multi(self, in_task):
        if in_task in self.multi_dict['done'] or in_task in self.multi_dict['run']:
            return
        if not in_task in self.task_demands:
            self.task_demands.append(in_task)
        if in_task in self.multi_dict['ready']:
            self.multi_dict['run'].append(in_task)
            self.multi_dict['ready'].remove(in_task)
            return
        else:
            self.insert_tower(in_task)
        return
    def insert_tower(self, in_task):
        for tw_idx in [0,1]:
            tw_fname = self.get_discv_fname(in_task[tw_idx])
            tw_action = in_task[tw_idx]
            if not exists(tw_fname) and not (tw_action in self.tower_demands):
                self.tower_demands.append(tw_action)
        return
    def refresh_command(self, calib_tag=False):
        self.update_dict("full")
        self.update_dict("single")
        self.update_dict("multi")
        self.update_demands()
        if calib_tag:
            self.update_calib()
        return
    def renew_command(self, calib_tag=False):
        self.update_dict("full")
        self.update_dict("single")
        self.update_dict("multi")
        self.update_demands()
        if calib_tag:
            self.update_calib()
        self.write_command("tower", 'w+')
        self.write_command("backwd", 'w+')
        self.write_command("forwd", 'a+')
        return
    def write_command(self, task_type, write_mode='a'):
        cmd_fname = dt.discv_command_fname
        str_split = '_'
        if task_type=="tower":
            cmd_fname = dt.tower_command_fname
            cmd_list = self.tower_demands
            str_split = ''
        elif task_type=="backwd":
            cmd_list = self.full_dict['run'] + self.full_dict['ready']
        elif task_type=="forwd":
            cmd_list_single = self.single_dict['run'] + self.single_dict['ready']
            cmd_list_multi = self.multi_dict['run']
            cmd_list = cmd_list_single + cmd_list_multi
        else:
            raise RuntimeError("    [Unit Manager] Wrong task_type!")
        original_stdout = sys.stdout
        with open(cmd_fname, write_mode) as f:
            sys.stdout = f
            for t in cmd_list:
                t_str = str_split.join(t)
                print('python code_main_discover_asynch.py '+task_type+' '+t_str)
        sys.stdout = original_stdout 
        if 'a' in write_mode:
            uniq_cmd = "sort -u "+cmd_fname+" > "+cmd_fname+"_tmp; mv "+cmd_fname+"_tmp "+cmd_fname
            os.system(uniq_cmd)
        return
    def collect_KLD(self):
        kld_recorder = {}
        for evals in [self.full_evals, self.single_evals, self.multi_evals]:
            for t in evals:
                t_causes, t_key = t.split('_')
                t_kld = evals[t][0]
                if not t_key in kld_recorder:
                    kld_recorder[t_key] = {}
                kld_recorder[t_key][t_causes] = t_kld
        return kld_recorder
    def update_dict(self, dict_name):
        if dict_name=="full":
            the_set = self.full_set
            the_dict = self.full_dict = {'null':[],'ready':[],'run':[], 'done':[]}
            the_evals = self.full_evals = {}
        elif dict_name=="single":
            the_set = self.single_set
            the_dict = self.single_dict = {'null':[],'ready':[],'run':[], 'done':[]}
            the_evals = self.single_evals = {}
        else: # multi
            the_set = self.multi_set
            the_dict = self.multi_dict = {'null':[],'ready':[],'run':[], 'done':[]}
            the_evals = self.multi_evals = {}
        for t in the_set:
            t_done_fname = self.get_discv_fname(t,True)
            t_run_fname = self.get_discv_fname(t,False)
            t_unit = []
            if exists(t_done_fname):
                the_dict['done'].append(t)
                t_unit = dt.load_obj(t_done_fname)
            elif exists(t_run_fname):
                the_dict['run'].append(t)
                t_unit = dt.load_obj(t_run_fname)
            if len(t_unit)>0:
                t_loss_rd = t_unit[1]
                idx = 1 # rmse
                t_evals = [t_loss_rd["brdg"]["KLD"][idx], t_loss_rd["brdg"]["recv"][idx], t_loss_rd["brdg"]["orig"][idx]]
                the_evals['_'.join(t)] = t_evals
            else:
                tw0_fname = self.get_discv_fname(t[0])
                tw1_fname = self.get_discv_fname(t[1])
                if exists(tw0_fname) and exists(tw1_fname):
                    tw0_finished_tag = dt.load_obj(tw0_fname)[-1][0]
                    tw1_finished_tag = dt.load_obj(tw1_fname)[-1][0]
                    if tw0_finished_tag and tw1_finished_tag:
                        the_dict['ready'].append(t)
                        continue
                the_dict['null'].append(t)
        if dict_name=="full" or dict_name=="single": # multi insert only
            for t in the_dict['null']:
                self.task_demands.append(t) # no matter finished=T/F
        return
    def update_demands(self):
        task_rm = []
        self.tower_demands = []
        for t in self.task_demands:
            # -- find dict_name --
            if t in self.full_set:
                dict_name = "full"
                the_dict = self.full_dict
            elif t in self.single_set:
                dict_name = "single"
                the_dict = self.single_dict
            else:
                dict_name = "multi"
                the_dict = self.multi_dict
            # -- process --
            if (t in the_dict['run']) or (t in the_dict['done']):
                task_rm.append(t)
                continue
            if dict_name=="full" or dict_name=="single":
                if t in the_dict['ready']:
                    task_rm.append(t)
                else:
                    self.insert_tower(t)
                continue
            if t in the_dict['ready']:    
                the_dict['run'].append(t)
                the_dict['ready'].remove(t)
            else:
                self.insert_tower(t)
        for t in task_rm:
            self.task_demands.remove(t)
        return
    def update_calib(self):
        self.single_dict.update({'reached':[]})
        self.multi_dict.update({'reached':[]})
        for t in self.full_dict['done']:
            t_causes = t[0]
            t_key = t[1]
            for c in t_causes:
                single_t = [c, t_key]
                self.single_dict['reached'].append(single_t)
            for L in range(2,len(t_causes)):
                for t_causes_comb in itertools.combinations(t_causes, L):
                    multi_t = [t_causes_comb, t_key]
                    self.multi_dict['reached'].append(multi_t)
        for t in (self.single_dict['run']):
            if not t in self.single_dict['reached']:
                self.single_dict['run'].remove(t)
        for t in (self.multi_dict['run']):
            if not t in self.multi_dict['reached']:
                self.multi_dict['run'].remove(t)
        check_set = self.single_dict['done'] + self.single_dict['run'] + self.multi_dict['done'] + self.multi_dict['run']
        for t in check_set:
            t_fname = self.get_discv_fname(t)
            if exists(t_fname):
                print("    [Unit Manager] Warning! Unreached unit exists:",t_fname)
        return
    def get_discv_fname(self, in_task, done_tag=[]):
        # done_tag: "op"=output, True=done, False=run, []=done/run
        if isinstance(in_task, str):
            # tower fname
            tw_fname, tw_op_fname = dt.get_discv_tower_fnames(in_task)
            if isinstance(done_tag, str) and done_tag=="op":
                return tw_op_fname
            return tw_fname
        else:
            # unit fname
            t_causes = in_task[0]
            t_key = in_task[1]
            if in_task in self.full_set:
                bk_tag = True
            elif not in_task in (self.single_set + self.multi_set):
                raise RuntimeError("    [Unit Manager] get_disc_fname: Wrong in_task", '_'.join(in_task))
            else:
                bk_tag = False
            run_fname, done_fname, op_fname = dt.get_discv_unit_fnames(t_key, t_causes, bk_tag)
            if isinstance(done_tag, str) and done_tag=="op":
                return op_fname
            if isinstance(done_tag, bool):
                if done_tag:
                    return done_fname
                else:
                    return run_fname
            if exists(done_fname):
                return done_fname
            else:
                return run_fname
            return


#TODO manager
class Preserve_Manager(object):
    def __init__(self, in_graph_obj, in_data_obj):
        super(Preserve_Manager, self).__init__()
        self.g_obj = in_graph_obj
        self.d_obj = in_data_obj
        n_nodes = self.d_obj.node_num
        self.node_keys = self.g_obj.node_keys
        kld_store = {}
        for k in self.node_keys:
            kld_store[k] = {}
        self.n_nodes = n_nodes
        self.kld_store = kld_store
        self.trim_store = {}
        self.target_key = self.d_obj.node_target # by default
        self.source_keys = self.d_obj.node_sources # by default
        self.u_obj = Unit_Manager(self.g_obj, self.target_key)
        self.show_box = Show_Box(self.u_obj, self.d_obj)
        self.show_rfn = Show_Box(self.u_obj, self.d_obj)
        self.selections = {'edge':[], 'gain':[]}
        self.predicters = {}
        self.save_fname = dt.manager_obj_fname
        print("[Preserve_Manager] created")
        return
    def save(self):
        saved_list = [[],[],[]]
        saved_list[0] = self.kld_store
        saved_list[1] = self.selections
        saved_list[2] = self.u_obj.get_weights([])
        dt.save_obj(saved_list, self.save_fname)
        return
    def load(self):
        if exists(self.save_fname):
            loaded_list = dt.load_obj(self.save_fname)
            self.kld_store = loaded_list[0]
            self.selections = loaded_list[1]
            self.u_obj.set_weights(loaded_list[2])
        return
    # ====================== Tower ======================
    def complete_tower(self):
        # single + full towers
        tw_tasks = []
        # --> single
        for s in self.u_obj.single_set:
            if not s[1] in tw_tasks:
                tw_tasks.append(s[1])
            if not s[0] in tw_tasks:
                tw_tasks.append(s[0])
        # --> full
        for s in self.u_obj.full_set:
            if not s[1] in tw_tasks:
                tw_tasks.append(s[1])
            if not s[0] in tw_tasks:
                tw_tasks.append(s[0])
        def myFunc(tw_str):
            return len(tw_str)
        tw_tasks.sort(key=myFunc)
        # running
        tw_tasks_run = []
        for t_str in tw_tasks:
            t_fname = self.u_obj.get_discv_fname(t_str)
            if exists(t_fname):
                t_finished_tag = dt.load_obj(t_fname)[-1][0]
                if t_finished_tag:
                    continue
            tw_tasks_run.append(t_str)
        self.tw_tasks = tw_tasks
        self.tw_tasks_run = tw_tasks_run
        # write tower commands
        cmd_fname = dt.tower_command_fname
        original_stdout = sys.stdout
        with open(cmd_fname, "w+") as f:
            sys.stdout = f
            for t in tw_tasks_run:
                print('python code_main_discover_asynch.py tower '+t)
        sys.stdout = original_stdout
        #self.show_box.display_evals(self.u_obj.single_set, [], "tower", "single")
        self.show_box.display_stats(self.u_obj.single_set, "single")
        self.show_box.display_stats(self.u_obj.full_set, "full")
        self.show_box.display_stats(self.u_obj.multi_set, "multi")
        return
    # ====================== Backprob ======================
    def complete_backprob(self): # run backprob without calibrate
        self.u_obj.renew_command()
        done_arr = self.u_obj.full_dict["done"]
        run_arr = self.u_obj.full_dict["run"]
        ready_arr = self.u_obj.full_dict["ready"]
        null_arr = self.u_obj.full_dict["null"]
        done_arr.extend(self.u_obj.single_dict["done"])
        run_arr.extend(self.u_obj.single_dict["run"])
        ready_arr.extend(self.u_obj.single_dict["ready"])
        null_arr.extend(self.u_obj.single_dict["null"])
        now = datetime.now()
        time_stamp = now.strftime("%m/%d %H:%M:%S")
        print("--- "+time_stamp+" ---")
        print("[Preserve_Manager] backprob renewal done.")
        print("    done tasks:", ', '.join(['_'.join(t) for t in done_arr]))
        print("    running tasks:", ', '.join(['_'.join(t) for t in run_arr]))
        print("    ready tasks:", ', '.join(['_'.join(t) for t in ready_arr]))
        print("    no-run tasks:", ', '.join(['_'.join(t) for t in null_arr]))
        return
    # ====================== Forward ======================
    def update_kld_store(self):
        kld_recoder = self.u_obj.collect_KLD()
        self.trim_store = {}
        def sum_path_kld(in_path):
            out_edges = []
            out_klds = []
            for i in range(len(in_path)-1):
                i_e = [in_path[i], in_path[i+1]]
                out_edges.append(i_e)
                i_e_kld = kld_recoder[i_e[1]][i_e[0]]
                out_klds.append(i_e_kld)
            return out_edges, out_klds
        for k in kld_recoder:
            self.kld_store[k] = kld_recoder[k]
            # -- trim store --
            det_arr = self.g_obj.nodes_detect_in(k)
            cuz_arr = self.g_obj.nodes_in(k)
            trim_arr = [d for d in det_arr if not d in cuz_arr]
            for t in trim_arr:
                trim_path = self.g_obj.path(t,k)
                if len(trim_path)>0: 
                # general type (cause->result with confounder) path_len>1
                    tm_edges, tm_klds = sum_path_kld(trim_path)
                    tm_spec_tags = [False for j in range(len(tm_edges))]
                elif t in self.source_keys: 
                # general type but cause=confounder=source, path_len>=1
                    for ss_key in [ss_k for ss_k in self.source_keys if not ss_k==t]:
                        trim_path = self.g_obj.path(ss_key,k)
                    tm_edges, tm_klds = sum_path_kld(trim_path)
                    tm_spec_tags = [False for j in range(len(tm_edges))]
                else: 
                # special type (confounder->result) path_1 & path_2 len>=1
                    trim_path_1 = []
                    trim_path_2 = []
                    for s in self.source_keys:
                        t_path = self.g_obj.path(s,t)
                        if trim_path_1==[] or len(t_path)<len(trim_path_1):
                            trim_path_1 = t_path
                        k_path = self.g_obj.path(s,k)
                        if trim_path_2==[] or len(k_path)<len(trim_path_2):
                            trim_path_2 = k_path
                    tm_1_res = sum_path_kld(trim_path_1) # tm_edges_1, tm_klds_1
                    tm_2_res = sum_path_kld(trim_path_2)
                    if sum(tm_1_res[1]) > sum(tm_2_res[1]):
                        tm_head = tm_1_res; tm_tail = tm_2_res
                    else:
                        tm_head = tm_2_res; tm_tail = tm_1_res
                    tm_edges = tm_head[0] + tm_tail[0]
                    tm_klds = tm_head[1] + [-val for val in tm_tail[1]]
                    tm_spec_tags = [False for j in tm_head[0]] + [True for j in tm_tail[0]]
                tm_pops = [0 for j in range(len(tm_edges))]
                self.trim_store[t+'_'+k] = {'tm_edges':tm_edges, 'tm_klds':tm_klds, 'tm_pops':tm_pops, 'special':tm_spec_tags}
        return
    def run_forward(self, the_sources=[]):
        if isinstance(the_sources, str):
            the_sources = [the_sources]
        if len(the_sources)==0:
            the_sources = self.source_keys
        # -- require full+single finished --
        self.u_obj.refresh_command()
        run_arr = self.u_obj.full_dict["run"]
        ready_arr = self.u_obj.full_dict["ready"]
        null_arr = self.u_obj.full_dict["null"]
        run_tag = True
        if len(null_arr)>0 or len(ready_arr)>0 or len(run_arr)>0:
            print("[Preserve_Manager] forward discovery: full tasks unfinished.")
            print("    full running tasks:", ', '.join(['_'.join(t) for t in run_arr]))
            print("    full ready tasks:", ', '.join(['_'.join(t) for t in ready_arr]))
            print("    full no-run tasks:", ', '.join(['_'.join(t) for t in null_arr]))
            run_tag = False
        run_arr = self.u_obj.single_dict["run"]
        ready_arr = self.u_obj.single_dict["ready"]
        null_arr = self.u_obj.single_dict["null"]
        if len(null_arr)>0 or len(ready_arr)>0 or len(run_arr)>0:
            print("[Preserve_Manager] forward discovery: single tasks unfinished.")
            print("    single running tasks:", ', '.join(['_'.join(t) for t in run_arr]))
            print("    single ready tasks:", ', '.join(['_'.join(t) for t in ready_arr]))
            print("    single no-run tasks:", ', '.join(['_'.join(t) for t in null_arr]))
            run_tag = False
        run_arr = self.u_obj.multi_dict["run"]
        if len(run_arr)>0: # only in run_arr demanged
            print("[Preserve_Manager] forward discovery: multi tasks unfinished.")
            print("    multi demanded tasks:", ', '.join(['_'.join(t) for t in run_arr]))
        if not run_tag:
            return run_tag
        # -- start --
        print("[Preserve_Manager] start forward discovery. ")
        self.update_kld_store()
        keys = self.node_keys
        # reset selections
        self.selections = {'edge':[], 'gain':[]}
        select_edges = self.selections['edge']
        select_gains = self.selections['gain']
        select_adj = {}
        gains_adj = {}
        views_kld = copy.deepcopy(self.kld_store) # 2D dictionary
        views_slt = {}
        action_required = []
        for k in keys:
            select_adj[k] = {i:0 for i in keys} # i->k selected
            gains_adj[k] = {i:float('inf') for i in keys} # k->i kld gain
            views_slt[k] = []
        # -- inner update --
        def get_view_gain(c_edge):
            e0, e1 = c_edge
            e1_kld = views_kld[e1]
            c_base = views_slt[e1].copy()
            c_base.sort()
            base_str = ''.join(c_base)
            c_arr = views_slt[e1].copy() +[e0]
            c_arr.sort()
            c_str = ''.join(c_arr)
            c_gain = e1_kld[c_str] - e1_kld[base_str]
            return c_gain
        def update_trim(slt_edge, candid_dict):
            # must after update_view()
            rm_tm_e = []
            for tm_e in self.trim_store:
                tm_edges = self.trim_store[tm_e]['tm_edges']
                tm_pops = self.trim_store[tm_e]['tm_pops']
                tm_spec_tags = self.trim_store[tm_e]['special']
                tm_len = len(tm_edges)
                for e_idx in range(tm_len):
                    if tm_pops[e_idx]==0 and tm_edges[e_idx]==slt_edge:
                        self.trim_store[tm_e]['tm_pops'][e_idx] = 1
                        tm_pops = self.trim_store[tm_e]['tm_pops']
                if sum(tm_pops)==tm_len:
                    rm_tm_e.append(tm_e)
                if not any(tm_spec_tags):
                    last_edge = tm_edges[-1]
                    last_pop = tm_pops[-1]
                    if last_pop==0 and last_edge[1]==slt_edge[1]:
                        self.trim_store[tm_e]['tm_klds'][-1] = get_view_gain(last_edge)
                    continue
                if tm_e.split('_')[1]!=slt_edge[1]:
                    continue
                split_len = tm_len-sum(tm_spec_tags)
                split_idx = split_len-1
                tm_klds = self.trim_store[tm_e]['tm_klds']
                for t in range(split_len, tm_len):
                    tm_klds[t] = -tm_klds[t]
                changed_tag = False
                
                for last_idx in [split_idx, -1]:
                    if tm_pops[last_idx]==0 and tm_edges[last_idx][1]==slt_edge[1]:
                        tm_klds[last_idx] = get_view_gain(tm_edges[last_idx])
                        changed_tag = True
                if changed_tag:
                    tm_dict_1 = {'edge':tm_edges[:split_len], 'kld':tm_klds[:split_len], 'pop':tm_pops[:split_len]}
                    tm_dict_2 = {'edge':tm_edges[split_len:], 'kld':tm_klds[split_len:], 'pop':tm_pops[split_len:]}
                    if sum(tm_dict_1['kld']) > sum(tm_dict_2['kld']):
                        tm_head = tm_dict_1; tm_tail = tm_dict_2
                    else:
                        tm_head = tm_dict_2; tm_tail = tm_dict_1
                    self.trim_store[tm_e]['tm_edges'] = tm_head['edge'] + tm_tail['edge']
                    self.trim_store[tm_e]['tm_klds'] = tm_head['kld'] + [-val for val in tm_tail['kld']]
                    self.trim_store[tm_e]['tm_pops'] = tm_head['pop'] + tm_tail['pop']
                    self.trim_store[tm_e]['special'] = [False for j in tm_head['edge']] + [True for j in tm_tail['edge']]
                    if tm_e.split('_') in candid_dict['det_edges']:
                        det_idx = candid_dict['det_edges'].index(tm_e.split('_'))
                        det_kld = sum(self.trim_store[tm_e]['tm_klds']) * trim_rate
                        candid_dict['det_gains'][det_idx] = det_kld
            for tm_e in rm_tm_e:
                self.trim_store.pop(tm_e)
            return candid_dict
        def update_view(slt_edge):
            s0, s1 = slt_edge
            s1_kld = views_kld[s1]
            causes_all = self.g_obj.nodes_in(s1)
            causes_slc = views_slt[s1]
            causes_left = [c for c in causes_all if not c in causes_slc and not c==s0]
            c_base = causes_slc.copy() + [s0]
            c_base.sort()
            base_str = ''.join(c_base)
            for c in causes_left:
                c_arr = c_base + [c]
                c_arr.sort()
                c_str = ''.join(c_arr)
                if c_str in s1_kld:
                    continue
                else:
                    action_required.append([c_str, s1])
                    avg_gain_arr = []
                    for the_str in s1_kld:
                        if c_str in the_str:
                            the_avg = (s1_kld[the_str]-s1_kld[base_str]) / (len(the_str)-len(c_str))
                            avg_gain_arr.append(the_avg)
                    avg_gain = np.mean(avg_gain_arr)
                    s1_kld[c_str] = s1_kld[base_str] + avg_gain
            views_slt[s1].extend([s0])
            return
        # -- run forward --
        reached_nodes = the_sources.copy()
        n_edges = self.g_obj.edge_num
        trim_rate = 1.11
        for n_round in range(n_edges): # n_round only for counting
            print("  # Round "+str(n_round+1)+":")
            candid_dict = {"edges":[], "gains":[], "det_edges":[], "det_gains":[]}
            # --> detect stage
            for b in reached_nodes:
                b_results = self.g_obj.nodes_out(b)
                candid_dict["edges"].extend([[b,r] for r in b_results if select_adj[r][b]==0])
            for e in candid_dict["edges"]:
                if len(views_slt[e[1]])==0:
                    e1_kld = views_kld[e[1]]
                    e_gain = e1_kld[e[0]]
                else:
                    e_gain = get_view_gain(e)
                candid_dict["gains"].append(e_gain)
            # --> for trimmed
            for b in reached_nodes:
                b_results = self.g_obj.nodes_detect_out(b)
                for r in b_results:
                    if [b,r] in candid_dict["edges"]:
                        candid_dict["det_edges"].append([b,r])
                        candid_dict["det_gains"].append(candid_dict["gains"][candid_dict["edges"].index([b,r])])
                    elif b+'_'+r in self.trim_store:
                        candid_dict["det_edges"].append([b,r])
                        trim_kld = self.trim_store[b+'_'+r]['tm_klds']
                        trim_kld = sum(trim_kld) * trim_rate
                        candid_dict["det_gains"].append(trim_kld)
            print("    Candid:", end=' ')
            for i in range(len(candid_dict["det_edges"])):
                e = candid_dict["det_edges"][i]
                e_gain = candid_dict["det_gains"][i]
                e_str = '_'.join(e)
                print('('+e_str+')%.4f' % (e_gain), end=' ')
                if (i+1)%8==0 and not i==len(candid_dict["det_edges"])-1:
                    print("\n           ", end=' ')
            # --> select stage
            slt_idx = np.argmin(candid_dict["gains"])
            slt_edge = candid_dict["edges"][slt_idx]
            slt_gain = candid_dict["gains"][slt_idx]
            slt_str = '_'.join(slt_edge)
            s0, s1 = slt_edge
            print("\n    Select:", end=' ')
            print('('+slt_str+')%.4f' % (slt_gain))
            # --> update view
            update_view(slt_edge)
            candid_dict = update_trim(slt_edge, candid_dict)
            select_adj[s1][s0] = 1
            select_edges.append(slt_edge)
            select_gains.append(slt_gain)
            print("    Trimmed:", end=' ')
            for i in range(len(candid_dict["det_edges"])):
                e = candid_dict["det_edges"][i]
                e_str = '_'.join(e)
                if e in candid_dict["edges"] or e_str in self.trim_store:
                    continue
                e_gain = candid_dict["det_gains"][i]
                print('('+e_str+')%.4f' % (e_gain), end=' ')
                if (i+1)%8==0 and not i==len(candid_dict["det_edges"])-1:
                    print("\n            ", end=' ')
            print("")
            # --> update reached nodes
            reached_nodes.append(s1)
            reached_nodes = list(np.unique(reached_nodes))
        # -- summarize --
        action_strs = []
        for a in action_required:
            self.u_obj.insert_multi(a)
            action_strs.append('_'.join(a))
        print("[Preserve_Manager] multi cause units required: "+', '.join(action_strs))
        demand_strs = ['_'.join(t) for t in self.u_obj.task_demands]
        print("    [task demands] "+', '.join(demand_strs))
        print("    [tower demands] "+', '.join(self.u_obj.tower_demands))
        self.u_obj.renew_command()
        return True
    # ====================== Prediction ======================
    def build_predicter(self, in_sources=[], if_refine=False, use_loaded=False):
        in_nodes = self.source_keys
        out_node = self.target_key
        if len(in_sources)>0:
            in_nodes = in_sources
        # --- check forwd finished ---
        slt_edges = self.selections['edge']
        rch_tags, rch_edges = self.g_obj.check_reach(in_nodes, out_node, slt_edges)
        slt_e_str = ', '.join(['_'.join(e) for e in slt_edges])
        rch_e_str = ', '.join(['_'.join(e) for e in rch_edges])
        if not all(rch_tags):
            print("[Preserve_Manager] create Predicter "+''.join(in_nodes)+'=>'+out_node+" fail!")
            _, required_edges = self.g_obj.check_reach(in_nodes, out_node)
            req_e_str = ', '.join(['_'.join(e) for e in required_edges])
            print("    required edges: " + req_e_str)
            print("    provided edges: " + slt_e_str)
            print("    provided effective edges: " + rch_e_str)
            raise RuntimeError("[Preserve_Manager] create Predicter fail!")
        # --- buildup ---
        if not slt_edges==rch_edges:
            print("[Preserve_Manager] Warning! create Predicter "+''.join(in_nodes)+'=>'+out_node+":")
            print("    provided edges: " + slt_e_str)
            print("    effective edges: " + rch_e_str)
        the_predr = Preserve_Predicter(self.d_obj, in_nodes, out_node, if_refine)
        the_predr.buildup(rch_edges, use_loaded)
        self.predicters[the_predr.predr_id] = the_predr
        if if_refine:
            self.refnr = the_predr
            self.show_rfn.set_predicter(the_predr)
        else:
            self.predr = the_predr
            self.show_box.set_predicter(the_predr)
        return


#TODO  Predictor
class Preserve_Predicter(object):
    def __init__(self, in_data_obj, in_sources=[], in_target='', if_refine=False):
        super(Preserve_Predicter, self).__init__()
        self.d_obj = in_data_obj
        self.node_keys = self.d_obj.node_keys
        self.n_nodes = self.d_obj.node_num
        if len(in_sources)>0:
            self.source_keys = in_sources
        else:
            self.source_keys = self.d_obj.node_sources
        if len(in_target)>0:
            self.target_key = in_target
        else:
            self.target_key = self.d_obj.node_target
        print("[Predicter] "+''.join(self.source_keys)+"=>"+in_target+" Created")
        if not self.d_obj.node_sources==self.source_keys:
            print("[Predicter] d_obj sources changed from "+''.joint(self.d_obj.node_sources)+' to '+''.join(self.source_keys)+'.')
            self.d_obj.node_sources = self.source_keys
        self.slt_edges = []
        self.ordered_edges = []
        self.ordered_nodes = []
        if if_refine:
            self.predr_id = "refnr_"+''.join(self.source_keys)+"~"+self.target_key
        else:
            self.predr_id = "predr_"+''.join(self.source_keys)+"~"+self.target_key
        self.log_id = "["+''.join(self.source_keys)+"~"+self.target_key+"]"
        #self.save_fname = dt.get_predicter_fname(self.predr_id)
    def buildup(self, in_slt_edges, use_loaded=False):
        ordered_edges, ordered_nodes = dt.depth_to_breadth(in_slt_edges, self.source_keys)
        self.slt_edges = in_slt_edges
        self.ordered_edges = ordered_edges # breadth first ordered edges
        self.ordered_nodes = ordered_nodes # fully reached ordered nodes
        # -- create nodes --
        nodes_store = {}
        self.nodes_store = nodes_store
        for k in self.node_keys:
            node = Status_Node(k, self.d_obj, self.predr_id)
            nodes_store[k] = node
        for i,e in enumerate(ordered_edges):
            nodes_store[e[1]].insert_cause(e[0])
        if use_loaded:
            for k in self.node_keys:
                nodes_store[k].load() # do not include buildup base or cause
            self.pred_routine = self.find_routine(self.target_key)
            # -- check achivement --
            _ = self.check_progress()
            # -- write commands --
            self.write_CMD()
            return
        else:
            for k in self.node_keys:
                nodes_store[k].save() # initialize all nodes
            self.pred_routine = self.find_routine(self.target_key)
        # -- check routine --
        for i in range(1,len(self.pred_routine)):
            key_r = self.pred_routine[i]
            node_r = self.nodes_store[key_r]
            key_c = self.pred_routine[i-1]
            assert key_c == node_r.flows_causes[0]
        # -- rebuild nodes --
        # predictor has no saved obj, only rebuild nodes
        # -- initialize entrance edges --
        idxs_1st_rch = []
        for n in self.ordered_nodes:
            for i,e in enumerate(self.ordered_edges):
                if (e[0] in self.source_keys) and (e[1]==n):
                    idxs_1st_rch.append(i)
                    break
        # -- buildup and check units --
        useless_flies = []
        for i,e in enumerate(self.ordered_edges):
            the_node = self.nodes_store[e[1]]
            if i in idxs_1st_rch:
                init_fname_set = dt.get_discv_unit_fnames(e[1], e[0])
                init_unit = un.Preserve_Unit(e[1], e[0], self.d_obj, init_fname_set)
                # discv_unit, AKA basic type unit only for loading
                init_unit.load()
                the_node.push_unit(init_unit)
                the_node.save()
                print("[Predicter]"+self.log_id+" "+"=>".join(e)+" entrances initialized.")
                continue
            ready_tag = the_node.check_ready(e[0])
            unit_fname_set = dt.get_train_unit_fnames(e[1], e[0], self.predr_id)
            unit_done_fname = unit_fname_set[1]
            if (not ready_tag) and exists(unit_done_fname):      
                next_tag = the_node.check_next(e[0])
                if next_tag:
                    print("[Predicter]"+self.log_id+" "+"=>".join(e)+" loading unit "+'_'.join(e))
                    the_unit = un.Preserve_Unit(e[1], e[0], self.d_obj, unit_fname_set) 
                    # discv_unit type, only for loading, but refin_unit also work
                    the_unit.load()
                    the_node.push_unit(the_unit)
                    the_node.save()
                else:
                    useless_flies.append(unit_done_fname)
        if len(useless_flies)>0:
            print("[Predicter]"+self.log_id+" "+"=>".join(e)+"Warning! unexpected files since previous edges not ready:")
            for fn in useless_flies:
                print("    "+fn)
        # -- check achivement --
        _ = self.check_progress()
        # -- write commands --
        self.write_CMD()
        return
    def reset_all(self):
        print("[Predicter] reset units as running.")
        for e in self.ordered_edges:
            the_node = self.nodes_store[e[1]]
            unit_fname_set = dt.get_train_unit_fnames(e[1], e[0], self.predr_id)
            the_unit = un.Preserve_Unit(the_node, self.nodes_store[e[0]], self.d_obj, unit_fname_set)
            the_unit.reset_running()
        print("[Predicter] reset nodes as empty.")
        for k in self.node_keys:
            self.nodes_store[k].reset_running()
        print("[Predicter] reset done.")
        # -- check achivement --
        self.buildup(self.slt_edges)
        return
    def check_progress(self):
        # -- check edges ready --
        e_done_arr = []
        for e in self.ordered_edges:
            e_ready = self.nodes_store[e[1]].check_ready(e[0])
            e_done_arr.append(e_ready)
        # -- check nodes ready --
        n_done_dict = {}
        for n in self.ordered_nodes:
            rch_tags = []
            for i,e in enumerate(self.ordered_edges):
                if e[1]==n:
                    rch_tags.append(e_done_arr[i])
            n_done_dict[n] = all(rch_tags) # all([])=True, sources included
        self.e_done_arr = e_done_arr
        self.n_done_dict = n_done_dict
        # --- update training progress ---
        train_dict = {'null':[],'ready':[],'run':[],'done':[]}
        for i,e in enumerate(self.ordered_edges):
            if e_done_arr[i]:
                train_dict['done'].append(e)
                continue
            if not n_done_dict[e[0]]:
                train_dict['null'].append(e)
            else:
                train_dict['ready'].append(e)
        for r in train_dict['ready']:
            r_node = self.nodes_store[r[1]]
            if not r_node.flows_causes[r_node.n_flows]==r[0]:
                continue
            r_conflict = False
            for u in train_dict['run']:
                if u[1]==r[1]:
                    r_conflict = True
                    break
            if not r_conflict:
                train_dict['run'].append(r)
        clear_ready = []
        for r in train_dict['ready']:
            if not r in train_dict['run']:
                clear_ready.append(r)
        train_dict['ready'] = clear_ready
        # -- save --
        e_done_str = ', '.join([str(r)[0] for r in e_done_arr])
        if all(e_done_arr):
            print("[Predicter]"+self.log_id+" Ready.")
        else:
            print("[Predicter]"+self.log_id+" Edges Progress ["+e_done_str+"]")
        n_edge = len(e_done_arr)
        info_str_arr = []
        for i in range(n_edge):
            e = self.ordered_edges[i]
            info_str_arr.append('['+str(i)+']'+'_'.join(e)+'('+str(e_done_arr[i])[0]+')')
        for s in range(int(np.ceil(n_edge/6))):
            info_str = " ".join(info_str_arr[s*6:min(n_edge,(s+1)*6)])
            print("        "+info_str)
        done_arr = []
        for e in train_dict['done']:
            e_rout = self.find_routine(e[0])
            e_rout.append(e[1])
            e_str = '_'.join(e_rout)
            done_arr.append(e_str)
        print("[Predicter]"+self.log_id+" Done Units: "+', '.join(done_arr))
        self.train_dict = train_dict
        return
    def write_CMD(self):
        cmd_fname = dt.train_command_fname
        original_stdout = sys.stdout
        with open(cmd_fname, 'w+') as f:
            sys.stdout = f
            for e in self.train_dict['run']:
                e_rout = self.find_routine(e[0])
                e_rout.append(e[1])
                e_str = '_'.join(e_rout)
                print('python code_main_train_asynch.py '+self.predr_id+' '+e_str)
        sys.stdout = original_stdout 
        print("[Predicter]"+self.log_id+" "+" Commands Update Done.")
        return
    # ---
    def find_routine(self, in_target_key, in_source_keys=[]):
        # recursively find routine to a source node
        if len(in_source_keys)==0:
            in_source_keys = self.source_keys
        if in_target_key in in_source_keys:
            return [in_target_key]
        node = self.nodes_store[in_target_key]
        out_routine = []
        for s in node.flows_causes:
            if s in in_source_keys:
                out_routine = [s, node.key]
                break
            else:
                s_routine = self.find_routine(s, in_source_keys)
                if len(s_routine)>0:
                    s_routine.append(node.key)
                    out_routine = s_routine
                    break
        return out_routine
    def save(self):
        for k in self.node_keys:
            self.nodes_store[k].save()
        return
    def __call__(self, in_streams, top_only, in_data=[]):
        entrance = in_streams[0]
        outlet = in_streams[-1]
        if in_data==[]:
            get_data_fun = self.d_obj.get_train_bundle
            #get_data_fun = self.d_obj.get_test_bundle
            input_x, input_c = get_data_fun(entrance.key) # or get_test
            output_x_recv, output_c_recv = get_data_fun(outlet.key)
            output_x_orig, output_c_orig = get_data_fun(outlet.key, False)
        else:
            input_x, input_c, output_x_recv, output_c_recv, output_x_orig, output_c_orig = in_data
        # ------
        dataset_list = input_x + input_c + output_x_recv + output_x_orig + output_c_recv
        dataset_tuple = tuple(dataset_list)
        data_slices = tf.data.Dataset.from_tensor_slices(dataset_tuple)
        data_slices = data_slices.batch(Batch_size)
        for step, batch_tuple in enumerate(data_slices):
            batch_in_x, batch_in_c = [batch_tuple[0]], [batch_tuple[1]] # bundle
            batch_out_x_recv = [batch_tuple[2]] # bundle
            batch_out_x_orig = [batch_tuple[3]]
            batch_out_c = [batch_tuple[-1]]
            cause_state = entrance.call_base_flow(batch_in_x, batch_in_c) # bundle
            for s in range(1, len(in_streams)-1): # exclude entrance and outlet
                cause_state = in_streams[s].call_cause_flow(cause_state)
            truth_list = [batch_out_x_recv, batch_out_x_orig, batch_out_c]
            if top_only:
                cause_state = outlet.call_cause_flow(cause_state) # bundle
                batch_losses = outlet.call_top_flow(cause_state, truth_list)
                if (step % 100 == 0 and verb >= 2) or (not in_data==[]):
                    print("    batch %d: recv_RMSE = %.4f, orig_RMSE = %.4f, mask = %.4f, recv_NSE = %.4f, orig_NSE = %.4f" %
                          (step,batch_losses[0],batch_losses[1],batch_losses[2],batch_losses[3],batch_losses[4]))
            else:
                cause_states_list = outlet.call_cause_flows(cause_state) # bundle list
                batch_losses_list = outlet.call_tops_flows(cause_states_list, truth_list)
                if (step % 100 == 0 and verb >= 2) or (not in_data==[]):
                    for L, batch_losses in enumerate(batch_losses_list):
                        print("    Output #"+str(L)+" batch %d: recv_RMSE = %.4f, orig_RMSE = %.4f, mask = %.4f, recv_NSE = %.4f, orig_NSE = %.4f" % 
                              (step,batch_losses[0],batch_losses[1],batch_losses[2],batch_losses[3],batch_losses[4]))
        if not top_only:
            layer_recv_arr = []
            for ls in range(len(outlet.losses)):
                 layer_recv_arr.append(outlet.losses[ls]['rmse']['recv'])
            layer_order = np.argsort(layer_recv_arr)
            print("    recv_rmse Ordered Layers: ")
            for ly in layer_order:
                print('        #'+str(ly)+' ('+str(layer_recv_arr[ly])[:5]+'), ',end='')
            min_ly = layer_order[0]
            print("        Layer #"+str(min_ly)+" selected.")
            avg_losses = outlet.losses[min_ly]
        else:
            print("        Top Only! Layer #"+str(outlet.n_flows-1)+" selected.")
            avg_losses = outlet.losses[0]
        return avg_losses
    def do_ext_predict(self, in_data_list, out_key, ext_fheader, top_only=True, cause_key=''):
        in_x, in_x_c, in_y, in_y_c, in_y_orig, in_y_c_orig = in_data_list
        # -------
        outlet = self.nodes_store[out_key]
        if outlet.n_flows==0:
            raise RuntimeError("[Predicter ext] external_data Node "+out_key+" unready to predict.")
        if cause_key=='':
            cause_key = outlet.flows_causes[outlet.n_flows-1]
        if not cause_key==outlet.flows_causes[outlet.n_flows-1] and top_only:
            print("[Predicter ext] required cause not top, cannot be top_only")
            top_only=False
        the_rout = self.find_routine(out_key)
        rt_len = len(the_rout)
        for t in range(1,rt_len-1):
            n = the_rout[t]
            if not self.n_done_dict[n]:
                raise RuntimeError("[Predicter ext] external_data Node "+n+" unready to predict.")
        print("[Predicter ext] external_data predict Node "+out_key+" on ", end='')
        if top_only:
            print("Top Level "+str(outlet.n_flows)+"-th.")
        else:
            print("All "+str(outlet.n_flows)+" Levels")
        # -- buildup streams --
        entrance = self.nodes_store[the_rout[0]]
        entrance.buildup("base")
        streams = [entrance]
        for t in range(1,rt_len): # including outlet
            t_node = self.nodes_store[the_rout[t]]
            t_node.buildup("cause")
            streams.append(t_node)
        outlet.buildup("tops")
        outlet_orig_opt_fname = outlet.output_fname
        ext_opt_fname = ext_fheader+outlet.key+"_"
        print("[Predicter ext] reset_output_pred: ext_opt_fname=", ext_opt_fname)
        outlet.reset_output_pred(ext_opt_fname)
        call_data_list = [in_x, in_x_c, in_y, in_y_c, in_y_orig, in_y_c_orig] # orig may be repeated recv
        # -- start --
        avg_losses = self.__call__(streams, top_only, call_data_list)
        outlet.output_fname = outlet_orig_opt_fname
        print("[Predicter ext] "+self.log_id+" predicted RMSE: mean recv = %.4f, mean orig = %.4f, mean mask = %.4f, mean recv_NSE = %.4f, mean orig_NSE = %.4f" % 
              (avg_losses['rmse']['recv'], avg_losses['rmse']['orig'], avg_losses['rmse']['mask'], avg_losses['NSE']['recv'], avg_losses['NSE']['orig']))
        print("[Predicter ext]"+self.log_id+" predicted MSE: mean recv = %.4f, mean orig = %.4f, mean mask = %.4f, mean recv_NSE = %.4f, mean orig_NSE = %.4f" % 
              (avg_losses['mse']['recv'], avg_losses['mse']['orig'], avg_losses['mse']['mask'], avg_losses['NSE']['recv'], avg_losses['NSE']['orig']))
        recv_values = pd.read_csv(ext_opt_fname+cause_key+'_recv.csv', delim_whitespace=True, header=0).values
        if isinstance(in_y, list):
            in_y = in_y[0]
        recv_mse, recv_rmse = both_loss_fn(in_y, recv_values)
        recv_NSE = nse_loss_fun(in_y, recv_values)
        print("[Predicter ext] output overall recv_mse = %.4f, recv_rmse = %.4f, recv_NSE = %.4f" % (recv_mse, recv_rmse, recv_NSE))
        return recv_values
    def do_predict(self, out_key='', top_only=True): # for test (can be training data)
        # Tip: training output by units, test output by nodes
        if out_key=='':
            out_key = self.target_key
        outlet = self.nodes_store[out_key]
        if outlet.n_flows==0:
            raise RuntimeError("[Predicter]"+self.log_id+" Node "+out_key+" unready to predict.")
        the_rout = self.find_routine(out_key)
        rt_len = len(the_rout)
        for t in range(1,rt_len-1):
            n = the_rout[t]
            if not self.n_done_dict[n]:
                raise RuntimeError("[Predicter]"+self.log_id+" Node "+n+" unready to predict.")
        print("[Predicter]"+self.log_id+" predict Node "+out_key+" on ", end='')
        if top_only:
            print("Top Level "+str(outlet.n_flows)+"-th.")
        else:
            print("All "+str(outlet.n_flows)+" Levels")
        # -- buildup streams --
        entrance = self.nodes_store[the_rout[0]]
        entrance.buildup("base")
        streams = [entrance]
        for t in range(1,rt_len): # including outlet
            t_node = self.nodes_store[the_rout[t]]
            t_node.buildup("cause") # buff_list built just before calling, no need of reset_cause()
            streams.append(t_node)
        outlet.buildup("tops")
        outlet.reset_output_pred()
        # -- start --
        avg_losses = self.__call__(streams, top_only)
        print("[Predicter]"+self.log_id+" predicted: mean recv_RMSE = %.4f, mean orig_RMSE = %.4f, mean mask = %.4f, mean recv_NSE = %.4f, mean orig_NSE = %.4f" % 
              (avg_losses[0], avg_losses[1], avg_losses[2], avg_losses[3], avg_losses[4]))
        return

#TODO Snode
class Status_Node(object):
    def __init__(self, my_key, in_d_obj, in_predr_id):
        super(Status_Node, self).__init__()
        self.key = my_key
        self.virtural_key = self.key
        self.d_obj = in_d_obj
        self.orig_tower = tw.Tower(self.key, self.d_obj)
        self.orig_tower.load()
        self.orig_tower.trainable = False
        self.orig_tower.tw_base.encoder.trainable = False
        self.orig_tower.tw_base.extracter.do_fun("disable")
        # ----
        self.states_stack = []
        self.virtural_stack = []
        self.IO_stack = []
        self.flows_causes = []
        self.n_flows = 0
        self.predr_id = in_predr_id
        fname_set = dt.get_train_node_fnames(self.key, in_predr_id)
        self.save_fname, self.output_fname = fname_set
        return
    def reset_output_pred(self, in_fname=''):
        tw_colnames = self.d_obj.get_names(self.key)[0]
        recv_header = ['RP_'+c[:3] for c in tw_colnames]
        orig_header = ['OP_'+c[:3] for c in tw_colnames]
        if len(in_fname) > 0:
            self.output_fname = in_fname
        self.meaners = []
        self.losses = []
        self.batch_counter = 0
        for l in range(self.n_flows):
            new_meaner = {}
            for loss_name in ['recv','orig','mask','NSE']:
                new_meaner[loss_name] = [AverageMeter(), AverageMeter()]
            self.meaners.append(new_meaner)
            self.losses.append({'mse':{}, 'rmse':{}, 'NSE':{}}) # if top_only, only use losses[-1]
            l_cause = self.flows_causes[l]
            recv_output_fname = self.output_fname+l_cause+'_recv.csv'
            orig_output_fname = self.output_fname+l_cause+'_orig.csv'
            init_stdout = sys.stdout
            with open(recv_output_fname, 'w') as f_recv:
                sys.stdout = f_recv
                for h in recv_header:
                    print("{0: <5} ".format(h), end='')
                print("")
            with open(orig_output_fname, 'w') as f_orig:
                sys.stdout = f_orig
                for h in orig_header:
                    print("{0: <5} ".format(h), end='')
                print("")
            sys.stdout = init_stdout
        pd.options.display.float_format = '{:,.2f}'.format
        return
    def insert_cause(self, in_key):
        self.flows_causes.append(in_key)
        return
    def check_ready(self, in_cause_key):
        c_idx = self.flows_causes.index(in_cause_key)
        if c_idx < self.n_flows:
            return True
        else:
            return False
    def check_next(self, in_cause_key):
        if self.flows_causes[self.n_flows]==in_cause_key:
            return True
        else:
            return False
    def push_unit(self, in_unit):
        # -- check unit validation --
        in_source = in_unit.tower_A.keys[0]
        assert self.flows_causes[self.n_flows]==in_source        
        w_A = copy.deepcopy(in_unit.A_adapter.get_weights([]))
        w_bridge = copy.deepcopy(in_unit.bridge.get_weights([]))
        w_B_out = copy.deepcopy(in_unit.B_top.get_weights([]))
        w_io_flow = [w_A, w_bridge, w_B_out]
        w_state = copy.deepcopy(in_unit.B_adapter.get_weights([]))
        self.IO_stack.append(w_io_flow)
        self.states_stack.append(w_state)
        self.n_flows += 1
        self.buildup("base")
        self.buildup("cause")
        self.buildup("tops")
        return
    def pop_unit(self): # rollback
        _ = self.IO_stack.pop()
        _ = self.states_stack.pop()
        self.n_flows -= 1
        self.buildup("base")
        self.buildup("cause")
        self.buildup("tops")
        return
    def get_top_flow(self, top_cause=''): # for unit using
        if self.n_flows==0:
            top_weights = self.orig_tower.tw_top.get_weights([])
        else:
            for ly in range(self.n_flows):
                ly_cause = self.flows_causes[ly]
                top_weights = self.IO_stack[ly][2]
                if ly_cause == top_cause: # default '' not matched
                    break
        return top_weights
    def get_cause_flow(self, in_key, top_cause='', enforce_full=False):
        print('[Node '+self.key+'] get cause_flow')
        def enforce_read():
            # despire of not done, but warning
            enfc_list = []
            stop_enfc_tag = False
            for i in range(self.n_flows, len(self.flows_causes)):
                i_cause = self.flows_causes[i]
                if i_cause == top_cause: # default '' not matched
                    break
                unit_run_fname, _, _ = dt.get_train_unit_fnames(self.key, i_cause, self.predr_id)
                unit_best_fname = unit_run_fname+"_best"
                if exists(unit_best_fname):
                    unit_run_fname = unit_best_fname
                if not exists(unit_run_fname):
                    stop_enfc_tag = True
                    print("    cause flow enforce not exists:",unit_run_fname)
                else:
                    if stop_enfc_tag:
                        print("    Warning! Found useless file:",unit_run_fname)
                    else:
                        print("    cause flow enforce reading...",unit_run_fname)
                        weights_kit, _, _, _ = dt.load_obj(unit_run_fname)
                        w_A = weights_kit["A_adapt"]
                        w_bridge = weights_kit["bridge"]
                        bridge_n_step = int(self.d_obj.get_unit_step(i_cause, self.key))
                        w_state = weights_kit["B_adapt"]
                        enfc_in_list = [w_A, w_bridge, bridge_n_step]
                        enfc_state_list = [w_state]
                        enfc_list.append([enfc_in_list, enfc_state_list])
            return enfc_list
        # ----
        cause_idx = self.flows_causes.index(in_key)
        cause_in_list = []        
        cause_state_list = []
        if self.n_flows-1 < cause_idx:
            if not enforce_full:
                raise RuntimeError("    cause flow #"+str(cause_idx)+" from Node "+in_key+" not exists!")
            else:
                enfc_res_list = enforce_read()
                if len(enfc_res_list)>0:
                    cause_in_list = enfc_res_list[0][0]
                    for res in enfc_res_list:
                        cause_state_list.append(res[1])
            return cause_in_list, cause_state_list
        # --- readable ---
        io_flow_set = self.IO_stack[cause_idx]
        cause_in_list.append(io_flow_set[0]) # A adapt
        cause_in_list.append(io_flow_set[1]) # bridge
        bridge_n_step = int(self.d_obj.get_unit_step(in_key, self.key))
        cause_in_list.append(bridge_n_step)
        print("    from :",self.flows_causes[cause_idx])
        print("    In_Flow: A_adp [len="+str(len(cause_in_list[0]))+"] brdg [len="+str(len(cause_in_list[1]))+"]")
        for ly in range(cause_idx+1, self.n_flows):
            ly_cause = self.flows_causes[ly]
            if ly_cause == top_cause: # default '' not matched
                break
            cause_state_list.append(self.states_stack[ly])
            print("    State_Flow: state "+str(ly)+" cause by", ly_cause,"len=",len(cause_state_list))
        if enforce_full:
            enfc_res_list = enforce_read()
            if len(enfc_res_list)>0:
                for res in enfc_res_list:
                    cause_state_list.append(res[1])
        return cause_in_list, cause_state_list
    def buildup(self, flow_type):
        state_seed = tf.random.normal([Batch_size,Dim_Repres,1], 0, 0.1)
        if flow_type=="base":
            self.states_flow = []
            for ly in range(0, self.n_flows):
                the_state = un.Adapt_State()
                the_state.trainable = False
                if ly==self.n_flows-1:
                    the_state.outlet_tag = True
                state_seed = the_state(state_seed)[0] # adpter only, no bridge
                the_state.set_weights(self.states_stack[ly])
                self.states_flow.append(the_state)
        elif flow_type=="cause":
            if self.n_flows==0:
                return
            self.cause_flow = {'in':[], 'state':[]}
            self.cause_buff = [] # only one bridge
            in_cause = self.flows_causes[0]
            io_flow_set = self.IO_stack[0]
            unit_n_step = self.d_obj.get_unit_step(in_cause, self.key)
            the_A = un.Adapt_State()
            the_A.trainable = False
            the_A.outlet_tag = True
            the_bridge = un.Bridge(unit_n_step)
            the_bridge.trainable = False
            brg_in_state = the_A(state_seed)[0]
            causal_seed, _ = the_bridge(brg_in_state,[])
            the_A.set_weights(io_flow_set[0])
            the_bridge.set_weights(io_flow_set[1])
            self.cause_flow['in'].append(the_A)
            self.cause_flow['in'].append(the_bridge)
            for ly in range(1, self.n_flows):
                the_state = un.Adapt_State()
                the_state.trainable = False
                if ly==self.n_flows-1:
                    the_state.outlet_tag = True
                causal_seed, _ = the_state(causal_seed)
                the_state.set_weights(self.states_stack[ly])
                self.cause_flow['state'].append(the_state)
        elif flow_type=="tops":
            if self.n_flows==0:
                raise RuntimeError('[Node '+self.key+'] tops flow buildup Err: No cause available!')
            self.tops_flows = []
            for l in range(self.n_flows):
                the_B_out = tw.Tower_Top(self.orig_tower.tw_base)
                the_B_out = self.orig_tower.copy_top(the_B_out, "_B_out")
                the_B_out.set_weights(self.IO_stack[l][2])
                self.tops_flows.append(the_B_out)
        elif flow_type=="virtural":
            if self.virtural_stack==[]:
                return
            self.virt_flow_obj = virt_flow(self.virtural_stack)
            self.virt_flow_obj.build_stream(state_seed)
        else: # calibrate 
            if self.calib_stack==[]:
                return
            self.calib_flow_obj = calib_flow(self.calib_stack)
            self.B_calib_adp = self.calib_flow_obj.build_stream(state_seed)
        return
    def reset_virtural(self):
        if self.virtural_stack==[]:
            return
        # reset buff from calling seeds
        if hasattr(self.virt_flow_obj, "buff_list"):
            for bf in range(len(self.virt_flow_obj.buff_list)):
                self.virt_flow_obj.buff_list[bf] = []
        else:
            self.buildup('virtural')
        return
    def call_virtural(self, in_state):
        if self.virtural_stack==[]:
            return in_state
        out_state = self.virt_flow_obj(in_state)
        return out_state
    def reset_calib(self):
        if self.calib_stack==[]:
            return
        # reset buff from calling seeds
        if exists(self.calib_flow_obj.buff_list):
            for bf in range(len(self.calib_flow_obj.buff_list)):
                self.calib_flow_obj.buff_list[bf] = []
        else:
            self.buildup('calib')
        return
    def call_calibrate(self, in_state):
        if self.calib_stack==[]:
            return in_state
        out_state = self.calib_flow_obj(in_state)
        return out_state
    def call_base_flow(self, input_x, input_c): # bundle
        # include call_state_flow
        base_out = self.orig_tower.call_base(input_x, input_c)
        out_state = base_out[2] # bundle
        for st in self.states_flow:
            out_state, _ = st(out_state)
        return out_state
    def call_state_flow(self, in_state): # bundle
        # Different from call_state in cause_flow!
        # built in buildup(base)!
        if self.n_flows==0:
            return in_state
        out_state = in_state
        for st in self.states_flow:
            out_state, _ = st(out_state)
        return out_state
    def call_cause_flow(self, in_state): # bundle
        out_state_list = self.call_cause_flows(in_state)
        out_state = out_state_list[-1]
        return out_state
    def call_cause_flows(self, in_state): # bundle
        if self.n_flows==0:
            raise RuntimeError('[Node '+self.key+'] n_flow==0 No Cause to call!')
        out_state = in_state
        adp_ly = self.cause_flow['in'][0]
        out_state, _ = adp_ly(out_state, [])
        brdg_ly = self.cause_flow['in'][1]
        out_state, self.cause_buff = brdg_ly(out_state, self.cause_buff)
        outputs = []
        outputs.append(out_state) # brdg output is outputs[0]
        for st_ly in self.cause_flow['state']: # may be [], when n_flows=1
            out_state, _ = st_ly(out_state)
            if not isinstance(out_state, list):
                added_form = [tf.identity(out_state)[:,:,0]]
            else:
                added_form = [out_state[0]]
            outputs.append(added_form)
        return outputs
    def call_top_flow(self, in_state, in_truth=[]): # bundle, list of 3 bundles
        output_losses = self.call_tops_flows([in_state], in_truth, True)
        return output_losses[0]
    def call_tops_flows(self, in_state_list, in_truth=[], top_only=False): # bundle list, list of 3 bundles
        if self.n_flows==0:
            raise RuntimeError('[Node '+self.key+'] tops flow calling Err: No cause available!')
        if self.n_flows < len(self.tops_flows):
            print('[Node '+self.key+'] Warning! Only '+str(self.n_flows)+'/'+str(len(self.tops_flows))+' cause flows loaded!')
        if top_only:
            the_tops = [self.tops_flows[self.n_flows-1]]
            out_causes = [self.flows_causes[self.n_flows-1]]
        else:
            the_tops = [self.tops_flows[l] for l in range(self.n_flows)]
            out_causes = self.flows_causes
        # ---
        if not in_truth==[]:
            true_x, true_x_orig, true_c = in_truth # bundle
        else:
            true_x, true_c = self.d_obj.get_batch_bundle_recv(self.key, self.batch_counter, Batch_size)
            true_x_orig, _ = self.d_obj.get_batch_bundle_orig(self.key, self.batch_counter, Batch_size)
            self.batch_counter += 1
        # ---
        true_x_expd, _, _, true_mask_expd, true_c_patch, _ = self.orig_tower.tw_base(true_x, true_c) # bundle
        true_x_orig = tf.cast(true_x_orig, 'float32')
        output_losses = []
        assert len(the_tops)==len(in_state_list)
        for t in range(len(the_tops)):
            t_state = in_state_list[t] # a bundle
            t_top = the_tops[t]
            # Training or Testing ????
            t_outputs = t_top(t_state, true_c_patch, true_x_expd, true_mask_expd)
            #t_outputs = t_top(t_state, true_c_patch)
            #############
            _, rls_output, out_x_res, out_x_merged, out_mask, out_mask_merged = t_outputs
            out_x_bundle = out_x_res[-1]
            out_x = out_x_merged[-1] # single, merged-bundle
            out_x_orig = self.orig_tower.descl_output(out_x_bundle)[0] # single
            out_mask_expd = out_mask[-1][0] # single
            recv_losses = both_loss_fn(true_x[0], out_x)
            orig_losses = both_loss_fn(true_x_orig[0], out_x_orig)
            mask_losses = binary_both_loss_fn(true_mask_expd[0], out_mask_expd)
            NSE_recv = nse_loss_fun(true_x[0], out_x)
            NSE_orig = nse_loss_fun(true_x_orig[0], out_x_orig)
            NSE_losses = [NSE_recv, NSE_orig]
            my_output_losses = [recv_losses[1], orig_losses[1], mask_losses[1], NSE_recv, NSE_orig]
            output_losses.append(my_output_losses)
            for idx in [0,1]:
                self.meaners[t]['recv'][idx].update(recv_losses[idx])
                self.meaners[t]['orig'][idx].update(orig_losses[idx])
                self.meaners[t]['mask'][idx].update(mask_losses[idx])
                self.meaners[t]['NSE'][idx].update(NSE_losses[idx])
            self.losses[t]['rmse']['recv'] = self.meaners[t]['recv'][1].avg.numpy()
            self.losses[t]['rmse']['orig'] = self.meaners[t]['orig'][1].avg.numpy()
            self.losses[t]['rmse']['mask'] = self.meaners[t]['mask'][1].avg.numpy()
            self.losses[t]['mse']['recv'] = self.meaners[t]['recv'][0].avg.numpy()
            self.losses[t]['mse']['orig'] = self.meaners[t]['orig'][0].avg.numpy()
            self.losses[t]['mse']['mask'] = self.meaners[t]['mask'][0].avg.numpy()
            self.losses[t]['NSE']['recv'] = self.meaners[t]['NSE'][0].avg.numpy()
            self.losses[t]['NSE']['orig'] = self.meaners[t]['NSE'][1].avg.numpy()
            self.output_pred(out_causes[t], out_x, out_x_orig)
        return output_losses
    def call_layer_flow(self, in_state, cause_in, cause_out):
        causes_arr = [''] + self.flows_causes
        layer_in = causes_arr.index(cause_in)
        layer_out = causes_arr.index(cause_out)
        if layer_in > layer_out:
            raise RuntimeError('[Node '+self.key+'] call_layer_flow invalid in_out '+layer_in+'_'+layer_out)
        out_state = in_state
        for ly in range(layer_in, layer_out):
            out_state = self.states_stack[ly](out_state)
        the_tw_top = tw.Tower_Top(self.orig_tower.tw_base)
        if layer_out==0:
            top_w = self.orig_tower.tw_top.get_weights([])
        else:
            top_w = self.IO_stack[layer_out][2]
        the_tw_top.set_weights(top_w)
        outputs = the_tw_top(out_state)
        return outputs
    def get_true_mask(self, input_x, input_c):
        true_mask_expd = self.orig_tower.tw_base([input_x], [input_c])[3] # convert to bundle
        true_mask_expd = true_mask_expd[0] # convert to single
        return true_mask_expd
    def save(self):
        saved_list = [[] for i in range(4)] # initial
        self.orig_tower.save()
        saved_list[0] = self.flows_causes
        saved_list[1] = self.n_flows
        saved_list[2] = self.IO_stack
        saved_list[3] = self.states_stack
        if not exists(self.save_fname):
            print('[Node '+self.key+'] saved file created!')
        dt.save_obj(saved_list, self.save_fname)
        return
    def load(self):
        if not exists(self.save_fname):
            raise RuntimeError('[Node '+self.key+'] saved file '+self.save_fname+' not exist! Run Predictor?')
        saved_list = dt.load_obj(self.save_fname)
        self.flows_causes = saved_list[0]
        self.n_flows = saved_list[1]
        self.IO_stack = saved_list[2]
        self.states_stack = saved_list[3]
        if not self.n_flows == len(self.IO_stack) == len(self.states_stack):
            raise RuntimeError('[Node '+self.key+'] loaded invalid! n_flows =',self.n_flows, 
                               'len(IO_stack) =', len(self.IO_stack), 'len(states_stack) =', len(self.states_stack))
        #self.buildup("base")
        #self.buildup("cause")
        return
    def reset_running(self):
        if exists(self.save_fname):
            saved_list = dt.load_obj(self.save_fname)
        else:
            saved_list = [[],[],[],[]]
            saved_list[0] = self.flows_causes
        saved_list[1] = 0
        saved_list[2] = []
        saved_list[3] = []
        dt.save_obj(saved_list, self.save_fname)
        print('    [Node '+self.key+'] reset running done.')
        return
    def output_pred(self, out_cause, data_recv_pred, data_orig_pred):
        recv_output_fname = self.output_fname+out_cause+'_recv.csv'
        orig_output_fname = self.output_fname+out_cause+'_orig.csv'
        recv_df = pd.DataFrame(data_recv_pred.numpy())
        orig_df = pd.DataFrame(data_orig_pred) 
        init_stdout = sys.stdout
        # -- recv --
        with open(recv_output_fname, 'a+') as f_recv:
            sys.stdout = f_recv
            print(recv_df.to_string(index=False, header=False))
        # -- orig --
        with open(orig_output_fname, 'a+') as f_orig:
            sys.stdout = f_orig
            print(orig_df.to_string(index=False, header=False))
        sys.stdout = init_stdout
        return


class virt_flow(keras.Model):
    def __init__(self, in_stack):
        super(virt_flow, self).__init__()
        self.w_stack = in_stack
        self.trainable = False
    def call(self, in_state):
        out_state = in_state
        for i, the_ly in enumerate(self.stream):
            out_state, return_buff = the_ly(out_state, self.buff_list[i])
            self.buff_list[i] = return_buff
        return out_state
    def build_stream(self, st_seed):
        self.stream = []
        self.buff_list = []
        for ly in range(len(self.w_stack)):
            if not isinstance(self.w_stack[ly], int):
                continue
            ly_param = self.w_stack[ly]
            ly_weights = self.w_stack[ly+1]
            if ly_param<=0:
                the_ly = un.Adapt_State()
                if ly_param<0:
                    the_ly.outlet_tag = True
                the_ly.trainable = False
            else:
                the_ly = un.Bridge(n_steps=ly_param, name="bridge_"+str(ly))
                the_ly.trainable = False
            st_seed, _ = the_ly(st_seed, [])
            the_ly.set_weights(ly_weights)
            self.stream.append(the_ly)
            self.buff_list.append([])
        return
    def get_weights(self, in_list):
        in_list = [[] for the_ly in self.stream]
        for i,the_ly in enumerate(self.stream):
            in_list[i] = the_ly.get_weights(in_list[i])
        return in_list
    def set_weights(self, in_list):
        for i,w in enumerate(in_list):
            self.w_stack[i] = copy.deepcopy(w)
            the_ly = self.stream[i]
            the_ly.set_weights(self.w_stack[i])
        return

class calib_flow(keras.Model):
    def __init__(self, in_stacks):
        super(calib_flow, self).__init__()
        self.w_stack = in_stacks
        self.trainable = True
    def call(self, in_state):
        out_state = in_state
        for i, the_ly in enumerate(self.stream):
            out_state, return_buff = the_ly(out_state, self.buff_list[i])
            self.buff_list[i] = return_buff
        return out_state
    def build_stream(self, st_seed):
        self.stream = []
        self.buff_list = []
        for ly in range(len(self.w_stack)):
            if not isinstance(self.w_stack[ly], int):
                continue
            if ly==0 or self.w_stack[ly-2]>0: # last is brdg, i.e. bottom of state stacks
                adp_ly = un.Adapt_State()
                #adp_ly = un.strong_Adapt()
                adp_ly.trainable = True
                st_seed, _ = adp_ly(st_seed, []) # init as Identity
                self.stream.append(adp_ly)
                self.buff_list.append([])
            #----
            ly_param = self.w_stack[ly]
            ly_weights = self.w_stack[ly+1]
            if ly_param<=0:
                the_ly = un.Adapt_State()
                if ly_param<0:
                    the_ly.outlet_tag = True
                the_ly.trainable = False
            else:
                the_ly = un.Bridge(n_steps=ly_param, name="bridge_"+str(ly))
                the_ly.trainable = False
            st_seed, _ = the_ly(st_seed, [])
            the_ly.set_weights(ly_weights)
            self.stream.append(the_ly)
            self.buff_list.append([])
        # the last adapter, for calib_vir stream
        adp_ly = un.Adapt_State()
        adp_ly.trainable = True
        st_seed, _ = adp_ly(st_seed, []) # init as Identity
        adp_ly.outlet_tag = True
        adp_ly.trainable = True
        self.stream.append(adp_ly)
        self.buff_list.append([])
        # the extra adapter, for B, stacked on B_tower states
        B_adp = un.strong_Adapt()
        B_adp.trainable = True
        st_seed, _ = B_adp(st_seed, []) # init as Identity
        B_adp.outlet_tag = True
        B_adp.trainable = True
        return B_adp
    def get_weights(self, in_list):
        in_list = [[] for the_ly in self.stream]
        for i,the_ly in enumerate(self.stream):
            in_list[i] = the_ly.get_weights(in_list[i])
        return in_list
    def set_weights(self, in_list):
        for i,w in enumerate(in_list):
            self.w_stack[i] = copy.deepcopy(w)
            the_ly = self.stream[i]
            the_ly.set_weights(self.w_stack[i])
        return


