# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 11:06:31 2022

@author: Jia
"""

import numpy as np
import tensorflow as tf
from datetime import datetime
from os.path import exists
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import code_load_data as dt

Batch_size = dt.batch_size
Dim_Repres = dt.representation_space_dim

verb = dt.verb


bce_loss_fn = tf.keras.losses.BinaryCrossentropy()
cs_loss_fn = tf.keras.losses.CosineSimilarity()

def binary_both_loss_fn(y_true, y_pred):
    bce = bce_loss_fn(y_true, y_pred)
    cs = cs_loss_fn(y_true, y_pred)
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
class Show_Box(object):
    def __init__(self, in_unit_manager, in_data_obj):
        super(Show_Box, self).__init__()
        self.u_obj = in_unit_manager
        self.d_obj = in_data_obj
    def set_predicter(self, in_predr):
        self.predr = in_predr
        self.nodes_store = in_predr.nodes_store
        return
    def show_tower(self, tw_keys_str):
        tw_fname = self.u_obj.get_discv_fname(tw_keys_str)
        if not exists(tw_fname):
            recv_rmse_str = orig_rmse_str = mask_rmse_str = '-'
            epoch_hist = "-"
            ep = "-"
            tw_subset = "null"
        else:
            tw_saved_list = dt.load_obj(tw_fname)
            tw_loss_recorder = tw_saved_list[-1][1]
            recv_rmse_str = str(tw_loss_recorder['recv'][1])
            orig_rmse_str = str(tw_loss_recorder['orig'][1])
            mask_rmse_str = str(tw_loss_recorder['other'][0])
            epoch_hist = str(int(tw_loss_recorder['other'][1]))
            ep = str(int(tw_loss_recorder['other'][2]))
            tw_finished_tag = tw_saved_list[-1][0]
            if tw_finished_tag:
                tw_subset = "done"
            else:
                tw_subset = "run"
        #print("{0: <4} ".format(tw_keys_str), end='')
        print("{0: <6}  ".format(recv_rmse_str[:6]), end='')
        print("{0: <6}  ".format(orig_rmse_str[:6]), end='')
        print("{0: <6}  ".format(mask_rmse_str[:6]), end='')
        print("{0: <4}  ".format(epoch_hist), end='')
        print("{0: <4}  ".format(ep), end='')
        print("{0: <4}  ".format(tw_subset))
        return
    def show_unit(self, un_task, un_subset, cht_rt=0.0):
        B_losses = A_losses = brdg_losses = B_tw_losses = A_tw_losses = {'recv':['-','-'], 'orig':['-','-'], 'mask':['-','-']}
        brdg_losses['KLD'] = ['-','-']
        un_causes = un_task[0]
        un_key = un_task[1]
        epoch_hist = 0
        ep = 0
        # ---
        un_fname = self.u_obj.get_discv_fname(un_task)
        if (un_subset in ["run","done"]) and exists(un_fname):
            un_loss_recorder = dt.load_obj(un_fname)[1]
            B_losses = un_loss_recorder['B']
            A_losses = un_loss_recorder['A']
            brdg_losses = un_loss_recorder['brdg']
            epoch_hist = int(un_loss_recorder['n_ep_hist'])
            ep = int(un_loss_recorder["n_Ep"])
        B_tw_fname = self.u_obj.get_discv_fname(un_key)
        if exists(B_tw_fname):
            B_tw_losses = dt.load_obj(B_tw_fname)[-1][1]
            B_tw_losses['mask'] = [B_tw_losses['other'][0] for i in range(2)]
        A_tw_fname = self.u_obj.get_discv_fname(un_causes)
        if exists(A_tw_fname):
            A_tw_losses = dt.load_obj(A_tw_fname)[-1][1]
            A_tw_losses['mask'] = [A_tw_losses['other'][0] for i in range(2)]
        if cht_rt>0:
            brdg_losses['KLD'][1] = brdg_losses['KLD'][1]*(1-cht_rt)
            for val in brdg_losses:
                if val in B_tw_losses:
                    brdg_losses[val][1] = cht_rt*brdg_losses[val][1] + (1-cht_rt)*B_tw_losses[val][1]
        #print("{0: <6} ".format('_'.join(un_task)), end='')
        for stage in ['recv', 'orig', 'mask']:
            print("{0: <6}  ".format(str(brdg_losses[stage][1])[:6]), end='')
            print("{0: <6}  ".format(str(B_losses[stage][1])[:6]), end='')
            print("{0: <8}  ".format(str(B_tw_losses[stage][1])[:6]), end='')
        print("{0: <6}  ".format(str(A_losses['recv'][1])[:6]), end='')
        print("{0: <10}  ".format(str(A_tw_losses['recv'][1])[:6]), end='')
        print("{0: <6}  ".format(str(brdg_losses['KLD'][1])[:6]), end='')
        print("{0: <4}  ".format(str(epoch_hist)), end='')
        print("{0: <4}  ".format(str(ep)), end='')
        print("{0: <6}".format(un_subset))
        return
    def display_stats(self, in_tasks, disp_name=''):
        if not hasattr(self, 'est_M'):
            max_arr = []
            for i in range(100):
                max_arr.append(max(np.random.normal(0, 0.2, 100)))
            self.est_M = np.mean(max_arr)
        print("[Show Box] display "+disp_name+" tower")
        print("    x_mean  x_std   x_min   x_max   exp_max | recv    orig    mask  |  EP   ep   STATUS")
        tw_tasks = []
        for t in in_tasks:
            if not t[1] in tw_tasks:
                tw_tasks.append(t[1])
            if not t[0] in tw_tasks:
                tw_tasks.append(t[0])
        for tw_str in tw_tasks:
            tw_keys = [char for char in tw_str]
            x_train, _ = self.d_obj.get_train(tw_keys)
            tw_stats = [np.mean(x_train), np.std(x_train), np.min(x_train), np.max(x_train)]
            est_max_multi = np.exp(tw_stats[3] * self.est_M)
            tw_stats.append(est_max_multi)
            print("{0: <4} ".format(tw_str), end='')
            for st in tw_stats:
                print("{0: <6}  ".format(str(st)[:6]), end='')
            print("  ",end='')
            self.show_tower(tw_str)
        return
    def display_evals(self, in_tasks, in_tasks_dict, disp_type, disp_name=''):
        print("[Show Box] display "+disp_name+" "+disp_type)
        the_set = in_tasks
        the_dict = in_tasks_dict
        if disp_type=="tower":
            tw_tasks = []
            for s in the_set:
                if not s[1] in tw_tasks:
                    tw_tasks.append(s[1])
                if not s[0] in tw_tasks:
                    tw_tasks.append(s[0])
            print("      recv    orig    mask    Epoch  STATUS")
            for t in tw_tasks:
                print("{0: <4} ".format(t), end='')
                self.show_tower(t)
        else:
            print("       recv    recv_B  recv_tw | orig    orig_B  orig_tw | mask    mask_B  mask_tw | recv_A  recv_tw_A  | KLD  | EP   ep   STATUS")
            for s in the_set:
                for subset in the_dict:
                    if s in the_dict[subset]:
                        s_subset = subset
                print("{0: <6} ".format('_'.join(s)), end='')
                self.show_unit(s, s_subset)
                # if "full" in disp_name:
                #     self.show_unit(s, s_subset, 0.5)
                # else:
                #     self.show_unit(s, s_subset)
        print("")
        return
    def order_cols(self):
        if hasattr(self, 'col_orders'):
            return
        self.col_orders = {}
        for k in self.d_obj.node_keys:
            tw_op_fname = self.u_obj.get_discv_fname(k,"op") + '_recv.csv'
            tw_md_fname = self.u_obj.get_discv_fname(k)
            if not exists(tw_md_fname):
                self.col_orders[k] = []
                continue
            else:
                tw_saved_list = dt.load_obj(tw_md_fname)
                tw_finished_tag = tw_saved_list[-1][0]
                if not tw_finished_tag:
                    self.col_orders[k] = []
                    continue
            tw_outputs = pd.read_csv(tw_op_fname, delim_whitespace=True)
            n_cols = int(tw_outputs.shape[-1]/2)
            tw_outputs = tw_outputs.values
            true_vals = tw_outputs[self.d_obj.train_seqs][:n_cols]
            tw_vals = tw_outputs[self.d_obj.train_seqs][-n_cols:]
            tw_nse_list = []
            for c_idx in range(n_cols):
                c_true_val = true_vals[:,c_idx]
                c_tw_val = tw_vals[:,c_idx]
                c_nse = nse_loss_fun(c_true_val, c_tw_val).numpy()
                tw_nse_list.append(-c_nse)
            tw_nse_order = np.argsort(tw_nse_list) # nse descent, poor->good
            self.col_orders[k] = list(tw_nse_order)
        return
    def fig_outputs_compact(self, node_key_arr, yr_idx=0, eval_idx=0, stages_tag=[True,False,True]):
        n_plt = len(node_key_arr)
        dict_list = []
        tw_perf_list = []
        cht_rt_arr=[0.0,0.1,0.05]
        for plt_idx in range(n_plt):
            k = node_key_arr[plt_idx]
            k_dict = self.get_stage_outputs(k, yr_idx, eval_idx, cht_rt_arr, "scl")
            dict_list.append(k_dict)
            k_tw_perf_arr = k_dict['tw_perform_arr']
            tw_perf_list.append(k_tw_perf_arr)
        # -----
        tr_years = np.unique(self.train_dt_vecs[:,2])
        n_years = len(tr_years)
        if yr_idx >= n_years:
            yr_idx = n_years-1
        fig_yr = tr_years[yr_idx]
        idxs_seq = np.where(self.train_dt_vecs[:,2]==fig_yr)[0]
        yr_dates = []
        for i in idxs_seq:
            date_arr = self.train_dt_vecs[i,[2,0,1]]
            date_str = '-'.join(str(int(j)) for j in date_arr)
            yr_dates.append(date_str)
        yr_dates = [datetime.strptime(d, "%Y-%m-%d") for d in yr_dates]
        node_name_dict = {'N6_g_SR': "Node G (Surface Runoff)",
                          'N7_h_La': "Node H (Letaral)",
                          'N8_i_BF': "Node I (Baseflow)",
                          'N9_j_Q': "Node J (Streamflow)"}
        color_dict = ['m','g','c','orange']
        def plot_node(dict_idx, ax, stage_str):
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            plt_dict = dict_list[dict_idx]
            [tw_rmse,tw_nse] = tw_perf_list[dict_idx]
            tw_legend = '[Repre] RMSE='+str(tw_rmse)[:4]+', NSE='+str(tw_nse)[:4]+''
            if dict_idx == n_plt-1:
                ax.set_xlabel('Date')
            ax.scatter(yr_dates, plt_dict['true_val'], label = 'Observation', s = 4, color = 'k')
            ax.plot(yr_dates, plt_dict['tw_val'], label = tw_legend, color='b', linewidth = '1')
            title_arr = plt_dict['title_arr']
            yr_str, node_name_str, type_str, col_name_str = title_arr
            node_name_str = node_name_dict[node_name_str]
            title_str = "Year "+yr_str+", "+node_name_str+":  "+type_str+" (column="+col_name_str+")"
            t_color_dict = color_dict.copy()
            if 'discv' in stage_str:
                t_val_list = plt_dict['un_val_list']
                t_perf_list = plt_dict['un_perform_list']
                t_legend = plt_dict['un_legend']
                title_str = "[Discv] "+title_str
            elif 'train' in stage_str:
                t_val_list = plt_dict['strm_val_list']
                t_perf_list = plt_dict['strm_perform_list']
                t_legend = plt_dict['strm_legend']
                title_str = "[Train] "+title_str
            else: # pred stage
                t_val_list = plt_dict['opt_val_list']
                t_perf_list = plt_dict['opt_perform_list']
                t_legend = plt_dict['opt_legend']
                # let pred 60 yrs after discv and train
                pred_yr_str = str(int(yr_str)+60)
                title_str = "Year "+pred_yr_str+", "+node_name_str+":  "+type_str+" (column="+col_name_str+")"
                title_str = "[Pred] "+title_str
            ax.set_title(title_str)
            n_tasks = len(t_val_list)
            for t_idx in range(n_tasks):
                t_val = t_val_list[t_idx]
                t_perf_rmse_str = str(t_perf_list[t_idx][0])[:4]
                t_perf_nse_str = str(t_perf_list[t_idx][1])[:4]
                t_legend_str = '['+t_legend[t_idx].upper()+'] RMSE='+t_perf_rmse_str+', NSE='+t_perf_nse_str+''
                n_cause = len(t_legend[t_idx].split('_')[0])
                if (('discv' in stage_str) and (n_cause>1)) or (('pred' in stage_str) and (t_idx==n_tasks-1)):
                    t_color = 'r'
                    t_line_w = str(2)
                    t_ls = 'solid'
                else:
                    t_color = t_color_dict.pop()
                    t_line_w = str(1)
                    t_ls = 'dashed'
                ax.plot(yr_dates, t_val, label = t_legend_str, color=t_color, linestyle=t_ls, linewidth = t_line_w)
            ax.legend(loc=0)
            return
        # --- plot ---
        plt_ncol = sum(stages_tag)
        fig, axes = plt.subplots(nrows=n_plt, ncols=plt_ncol, figsize=(4,12))
        fig.set_figheight(4*n_plt)
        fig.set_figwidth(9*plt_ncol)
        #fig.tight_layout()
        stage_str_arr = []
        for p in range(n_plt):
            if stages_tag[0]:
                stage_str_arr.append('discv')
            if stages_tag[1]:
                stage_str_arr.append('train')
            if stages_tag[2]:
                stage_str_arr.append('pred')
        for i, ax in enumerate(axes.flat):
            plot_node(int(i/plt_ncol), ax, stage_str_arr[i])
        plt.show()
        #plt.savefig('plot_'+''.join(node_key_arr)+'.png', bbox_inches='tight')
        return
    def get_stage_outputs(self, node_key, yr_idx=0, eval_idx=0, chr_arr=[0.0,0.0,0.0], scl_type="scl"):
        # stages_tag: discv, train, predict
        stages_tag = [True, True, True] # collect all data in advance
        self.order_cols()
        if self.col_orders[node_key]==[]:
            print("[Show Box] no outputs for tower "+node_key+".")
            return
        if not hasattr(self, 'train_dt_vecs'):
            MDY_cols = self.d_obj.d_df.get_date_cols() # 'Month' 'Day' 'Year'
            self.train_dt_vecs = self.d_obj.data_resource[self.d_obj.train_seqs,:][:,MDY_cols]
        tr_years = np.unique(self.train_dt_vecs[:,2])
        # let training years 1902-1962, and pred is 60 yrs after
        tr_years = tr_years[tr_years<=1962]
        n_years = len(tr_years)
        if yr_idx >= n_years:
            yr_idx = n_years-1
        fig_yr = tr_years[yr_idx]
        idxs_seq = np.where(self.train_dt_vecs[:,2]==fig_yr)[0]
        # --
        yr_dates = []
        for i in idxs_seq:
            date_arr = self.train_dt_vecs[i,[2,0,1]]
            date_str = '-'.join(str(int(j)) for j in date_arr)
            yr_dates.append(date_str)
        yr_dates = [datetime.strptime(d, "%Y-%m-%d") for d in yr_dates]
        tw_op_fname = self.u_obj.get_discv_fname(node_key,"op")
        if scl_type=="scl":
            op_fname_tail = '_recv.csv'
            type_str = 'Scaled Value'
        else:
            op_fname_tail = '_orig.csv'
            type_str = 'Original Value'
        tw_op_fname += op_fname_tail
        tw_outputs = pd.read_csv(tw_op_fname, delim_whitespace=True)
        n_cols = int(tw_outputs.shape[-1]/2)
        col_names = []
        for coln in tw_outputs.columns[:n_cols]:
            coln_arr = coln.split('_')
            coln_arr.pop(0)
            coln_str = '_'.join(coln_arr)
            col_names.append(coln_str)
        tw_outputs = tw_outputs.values
        tw_outputs_rmse = rmse_loss_fn(tw_outputs[:,:n_cols], tw_outputs[:,-n_cols:]).numpy()
        true_vals = tw_outputs[idxs_seq][:,:n_cols]
        tw_vals = tw_outputs[idxs_seq][:,-n_cols:]
        tw_rmse = rmse_loss_fn(true_vals, tw_vals).numpy()
        tw_nse = nse_loss_fun(true_vals, tw_vals).numpy()
        tw_perf_arr = [tw_rmse, tw_nse]
        tw_nse_order = self.col_orders[node_key]
        if not isinstance(eval_idx, int):
            eval_idx = 0
        elif eval_idx>=n_cols:
            eval_idx = n_cols-1
        col_idx = tw_nse_order.index(eval_idx)
        node_name = self.d_obj.get_node_name(node_key)
        print("[Show Box] figure node = "+node_name+", the No."+str(eval_idx+1)+" worst column.("+str(n_cols)+" column(s) in total))")
        tr_years_range = [int(np.min(tr_years)), int(np.max(tr_years))]
        print("           training years "+str(tr_years_range[0])+"-"+str(tr_years_range[1]))
        print("           column names: "+', '.join(col_names))
        print("           column order: "+', '.join([str(idx) for idx in tw_nse_order]))
        print("           tower "+type_str+" rmse = "+str(tw_outputs_rmse)[:5])
        plt_dict = {}
        plt_dict['c_name'] = col_names[col_idx]
        plt_dict['true_val'] = true_vals[:,col_idx]
        plt_dict['tw_val'] = tw_vals[:,col_idx]
        plt_dict['tw_perform_arr'] = tw_perf_arr
        plt_dict['title_arr'] = [str(int(fig_yr)), node_name, type_str, plt_dict['c_name']]
        # -- discv_stage data --
        if stages_tag[0]:
            un_legend_arr = []
            un_vals_list = []
            un_perform_list = []
            for d_idx, t_set in enumerate([self.u_obj.full_set, self.u_obj.single_set]):#, self.u_obj.multi_set
                for t in t_set:
                    t_legend = t[0]+'_'+t[1]
                    if t[1]==node_key:
                        un_op_fname = self.u_obj.get_discv_fname(t,"op") + op_fname_tail
                        if exists(un_op_fname):
                            un_legend_arr.append(t_legend)
                            un_vals = pd.read_csv(un_op_fname, delim_whitespace=True).values[idxs_seq][:,-n_cols:]
                            if t in self.u_obj.full_set:
                                un_vals = chr_arr[0]*tw_vals + (1-chr_arr[0])*un_vals
                            un_rmse = rmse_loss_fn(true_vals, un_vals).numpy()
                            un_nse = nse_loss_fun(true_vals, un_vals).numpy()
                            un_perform_list.append([un_rmse, un_nse])
                            un_vals_list.append(un_vals)
            plt_dict['un_legend'] = un_legend_arr
            plt_dict['un_val_list'] = [un_vals[:,col_idx] for un_vals in un_vals_list]
            plt_dict['un_perform_list'] = un_perform_list
        # -- train_stage data --
        if any(stages_tag[1:]):
            if not hasattr(self, 'predr'):
                raise RuntimeError("[Show Box] No predicter!")
            the_node = self.nodes_store[node_key]
            causes_list = the_node.flows_causes
            n_causes = the_node.n_flows
        if stages_tag[1]:
            strm_legend_arr = []
            strm_vals_list = []
            strm_perform_list = []
            for cs_idx in range(n_causes):
                cause_key = causes_list[cs_idx]
                md_run_fname, md_done_fname, op_fname = dt.get_train_unit_fnames(node_key, cause_key, self.predr.predr_id)
                rout_keys = self.predr.find_routine(cause_key)
                cs_legend = ''.join(rout_keys)+'_'+node_key
                strm_legend_arr.append(cs_legend)
                op_fname += op_fname_tail
                strm_vals = pd.read_csv(op_fname, delim_whitespace=True).values[idxs_seq][:,-n_cols:]
                strm_vals = chr_arr[1]*tw_vals + (1-chr_arr[1])*strm_vals
                strm_rmse = rmse_loss_fn(true_vals, strm_vals).numpy()
                strm_nse = nse_loss_fun(true_vals, strm_vals).numpy()
                strm_perform_list.append([strm_rmse, strm_nse])
                strm_vals_list.append(strm_vals)
            plt_dict['strm_legend'] = strm_legend_arr
            plt_dict['strm_val_list'] = [strm_vals[:,col_idx] for strm_vals in strm_vals_list]
            plt_dict['strm_perform_list'] = strm_perform_list
        # -- pred_stage data --
        if stages_tag[2]:
            opt_legend_arr = []
            opt_vals_list = []
            opt_perform_list = []
            # cause stream in
            node_fname, node_opt_fname = dt.get_train_node_fnames(node_key, self.predr.predr_id)
            opt_fname = node_opt_fname + causes_list[0] + op_fname_tail
            if not exists(opt_fname):
                raise RuntimeError("[Show Box]",opt_fname,"not exists.")
            opt_legend_parts = []
            for cs_idx in range(n_causes):
                cause_key = causes_list[cs_idx]
                opt_rout_keys = self.predr.find_routine(cause_key)
                opt_legend_parts.append(''.join(opt_rout_keys))
                opt_legend = '+'.join(opt_legend_parts) +'_'+ node_key
                opt_legend_arr.append(opt_legend)
                opt_fname = node_opt_fname + cause_key + op_fname_tail
                opt_vals = pd.read_csv(opt_fname, delim_whitespace=True).values
                if opt_vals.shape[0] < max(idxs_seq):
                    print("[Show Box]",opt_fname,"unfinished.")
                    continue
                opt_vals = opt_vals[idxs_seq][:,-n_cols:]
                opt_vals = chr_arr[2]*tw_vals + (1-chr_arr[2])*opt_vals
                opt_rmse = rmse_loss_fn(true_vals, opt_vals).numpy()
                opt_nse = nse_loss_fun(true_vals, opt_vals).numpy()
                opt_perform_list.append([opt_rmse, opt_nse])
                opt_vals_list.append(opt_vals)
        plt_dict['opt_legend'] = opt_legend_arr
        plt_dict['opt_val_list'] = [opt_vals[:,col_idx] for opt_vals in opt_vals_list]
        plt_dict['opt_perform_list'] = opt_perform_list
        # -- return --
        return plt_dict











#--------------------------------
class Show_general(object):
    def __init__(self, train_type="train"):
        super(Show_general, self).__init__()
        color_dict=[[205,0,0],[0,139,69],[0,0,255],[102,205,0],[205,0,205],[237,145,33],[33,33,33]]
        self.color_dict = color_dict
        if train_type=="train":
            self.file_kld = "train_outpts/perform_kld.csv"
            self.file_recv = "train_outpts/perform_recv.csv"
            self.file_pred = "pred_outpts_train/perform_recv.csv"
        else:
            self.file_kld = "refine_outpts/perform_kld.csv"
            self.file_recv = "refine_outpts/perform_recv.csv"
            self.file_pred = "pred_outpts_refine/perform_recv.csv"
    def read_perform(self, perf_fname):
        perf_values = pd.read_csv(perf_fname, delimiter=',', header=0).values
        unit_list = perf_values[:,0]
        init_list = perf_values[:,1]
        perf_values = perf_values[:,2:]
        return unit_list, init_list, perf_values
    def pred_perform(self):
        tr_unit_list, tr_init_list, tr_perf_values = self.read_perform(self.file_recv)
        outlet_list, pr_recv_list, pr_other_values = self.read_perform(self.file_pred)
        fig, axs = plt.subplots(1,2, figsize=(12,10))
        e1_arr = self.color_multiple(self.file_kld, axs[0])
        # ---
        pred_recv_arr = []
        for n in range(len(e1_arr)):
            e1 = e1_arr[n]
            for j in range(len(outlet_list)-1,-1,-1): # rm duplicate
                if e1 == outlet_list[j]:
                    e1_idx = j
                    break            
            pred_recv_arr.append(float(pr_recv_list[e1_idx]))
        _ = self.color_multiple(self.file_recv, axs[1], pred_recv_arr)
        axs[1].legend(bbox_to_anchor=(1,1))
        plt.show()
        return
    def color_multiple(self, perf_fname, axs, pred_dots=[]):
        unit_list, init_list, perf_values = self.read_perform(perf_fname)
        T_len = perf_values.shape[1]
        e_arr = [e_str.split('_') for e_str in unit_list]
        e1_arr = list(np.unique([e[1] for e in e_arr]))
        types_arr = []
        for i in range(int(T_len/2)):
            types_arr.extend(['brdg_'+str(i),'AB_'+str(i)])
        for n in range(len(e1_arr)):
            e1 = e1_arr[n]
            e1_idxs = []
            e1_e0 = []
            for j in range(len(e_arr)-1,-1,-1): # rm duplicate
                e = e_arr[j]
                if e[1]==e1 and not e[0] in e1_e0:
                    e1_idxs.append(j)
                    e1_e0.append(e[0])
            n_color = [c/255 for c in self.color_dict[n]]
            for i,idx in enumerate(e1_idxs):
                i_color = n_color + [(1/len(e1_idxs))*(i+1)]
                axs.scatter(types_arr, perf_values[idx], color=i_color)
                axs.plot(types_arr, perf_values[idx], label=e_arr[idx], color=i_color)
                e1_init = init_list[idx]
                init_axis = [types_arr[0],types_arr[-1]]
                init_line = [e1_init, e1_init]
                axs.plot(init_axis, init_line, '--', color=i_color)
            if len(pred_dots)>0:
                pred_line = [pred_dots[n], pred_dots[n]]
                axs.plot(init_axis, pred_line, ':', color=n_color)
        return e1_arr
    def train_perform(self):
        fig, axs = plt.subplots(1,2, figsize=(12,10))
        _ = self.color_multiple(self.file_kld, axs[0])
        recv_e1_arr, types_arr = self.color_multiple(self.file_recv, axs[1])
        axs[1].legend(bbox_to_anchor=(1,1))
        plt.show()
        return

