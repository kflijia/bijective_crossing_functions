# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 10:49:54 2021

@author: Jia
"""

import numpy as np
import tensorflow as tf
#from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
#import fnmatch
import os
import sys
from os.path import exists
#from datetime import datetime
#import matplotlib.dates as mdates
#import matplotlib.pyplot as plt

global verb
verb = 3 # 0, 1, 2

# for biject encoder
expd_x_dim = 12
sampling_space_dim = 16

representation_space_dim = 16

batch_size = 64


#    Conditions:
#    Wind speed:WINDM/s
#    Relative humidity: RHmd
#    Temperature: TMP_MXdgC, TMP_MNdgC
#    Solar radiation: SOLARMJ/m2
#    Precipitation: PRECIPmm
#    
#    Others: 
#    Evapotranspiration: ETmm
#    snow pack: SNOmm
#    soil water: SW_ENDmm
#    aquifer: SA_STmm
#    surface runoff: SURQ_CNTmm
#    lateral: LATQCNTmm
#    baseflow: GW_Qmm
#    streamflow: Q_pred_mm
#    (aquifer) PERCmm: Water that percolates through the root zone
#    (aquifer) REVAPmm: Water in the shallow aquifer flowing back to the root zone in response to moisture deficit
#    (baseflow) GW_RCHGmm: groundwater recharge
#    (baseflow)TLOSSmm: transmission loss
#    (aquifer) DA_RCHGmm: deep aquifer recharge
#    (surface runoff) DAILYCN: daily curve number. One parameter that SWAT uses to calculate surface runoff. 
#    (surface runoff) LAI: leaf area index
#    (streamflow) QTILEmm: tile drainage flow. It is not streamflow and it should be separate columns from Q_pred_mm. 

data_source_fname = "SWAT_data/short_data_readable.csv"

data_obj_fname = "obj_data.pickle"
graph_obj_fname = "obj_graph.pickle"
manager_obj_fname = "obj_manager.pickle"
#predictor_obj_fname = "obj_predictor.pickle" 

tower_model_fname_header = "discv_towers/tower_"
tower_res_fname_header = "discv_outpts/res_tower_"
backprob_unit_fname_header = "discv_units/unit_backwd_"
forward_unit_fname_header = "discv_units/unit_forwd_"
backprob_res_fname_header = "discv_outpts/res_backwd_"
forward_res_fname_header = "discv_outpts/res_forwd_"

calib_unit_fname_header = "calib_units/unit_"
calib_snode_fname_header = "calib_nodes/node_"
calib_res_fname_header = "calib_outpts/res_"

train_unit_fname_header = "train_units/unit_"
train_snode_fname_header = "train_nodes/node_"
train_res_fname_header = "train_outpts/res_"

refine_unit_fname_header = "refine_units/unit_"
refine_snode_fname_header = "refine_nodes/node_"
refine_res_fname_header = "refine_outpts/res_"

pred_trn_fname_header = "pred_outpts_train/res_"
pred_rif_fname_header = "pred_outpts_refine/res_"
finished_unit_fname_tail = "_Done"
unfinished_unit_fname_tail = "_Run"


tower_command_fname = "CMDbuffer_tower.txt"
discv_command_fname = "CMDbuffer_discover.txt"
train_command_fname = "CMDbuffer_train.txt"



def save_obj(in_obj, obj_fname):
    obj_file = open(obj_fname, "wb")
    pickle.dump(in_obj, obj_file)
    obj_file.close()

def load_obj(obj_fname):
    obj_file = open(obj_fname, "rb")
    out_obj = pickle.load(obj_file)
    obj_file.close()
    return out_obj

# all input nodes are in key or keys_list
def get_discv_tower_fnames(in_keys):
    if isinstance(in_keys,list):
        str_keys = ''.join(in_keys)
    else:
        str_keys = str(in_keys)
    tw_fname = tower_model_fname_header+str_keys
    output_fname = tower_res_fname_header+str_keys
    return tw_fname, output_fname
def get_discv_unit_fnames(in_key, in_causes, bk_tag=[]):
    if bk_tag==[]:
        fname_triple = get_discv_unit_fnames(in_key, in_causes, True)
        if not exists(fname_triple[0]):
            fname_triple = get_discv_unit_fnames(in_key, in_causes, False)
        return fname_triple
    str_causes = ''.join(in_causes)
    done_ftail = finished_unit_fname_tail
    run_ftail = unfinished_unit_fname_tail
    if bk_tag:
        fheader = backprob_unit_fname_header
        res_fheader = backprob_res_fname_header
    else:
        fheader = forward_unit_fname_header
        res_fheader = forward_res_fname_header
    unit_run_fname = fheader+in_key+'_'+str_causes+run_ftail
    unit_done_fname = fheader+in_key+'_'+str_causes+done_ftail
    output_fname =  res_fheader+in_key+'_'+str_causes
    return unit_run_fname, unit_done_fname, output_fname
# ----
# def get_predicter_fname(in_predr_id):
#     id_str = in_predr_id.split('_')[1]
#     fname_parts = predictor_obj_fname.split('.')
#     predr_fname = fname_parts[0]+'_'+id_str+'_'+fname_parts[1]
#     return predr_fname
def get_calib_unit_fnames(in_key, in_cause, in_predr_id):
    in_out_str = in_cause+'_'+in_key
    id_str = in_predr_id.split('_')[1]
    run_ftail = unfinished_unit_fname_tail
    done_ftail = finished_unit_fname_tail
    u_fheader = calib_unit_fname_header
    r_fheader = calib_res_fname_header
    unit_fname_run = u_fheader+id_str+'_'+in_out_str+run_ftail
    unit_fname_done = u_fheader+id_str+'_'+in_out_str+done_ftail
    output_fname = r_fheader+id_str+'_'+in_out_str
    return unit_fname_run, unit_fname_done, output_fname
def get_train_unit_fnames(in_key, in_cause, in_predr_id):
    in_out_str = in_cause+'_'+in_key
    type_str, id_str = in_predr_id.split('_')
    run_ftail = unfinished_unit_fname_tail
    done_ftail = finished_unit_fname_tail
    if "refnr" in type_str:
        u_fheader = refine_unit_fname_header
        r_fheader = refine_res_fname_header
    else:
        u_fheader = train_unit_fname_header
        r_fheader = train_res_fname_header
    unit_fname_run = u_fheader+id_str+'_'+in_out_str+run_ftail
    unit_fname_done = u_fheader+id_str+'_'+in_out_str+done_ftail
    output_fname = r_fheader+id_str+'_'+in_out_str
    return unit_fname_run, unit_fname_done, output_fname
def train2discv_unit_fnames(in_unit_fname_run):
    if "refine" in in_unit_fname_run:
        u_fheader = refine_unit_fname_header
    else:
        u_fheader = train_unit_fname_header
    fname_run = in_unit_fname_run[len(u_fheader):]
    parts_arr = fname_run.split('_')
    in_cause = parts_arr[1]
    in_key = parts_arr[2]
    return get_discv_unit_fnames(in_key, [in_cause])
def get_train_node_fnames(in_key, in_predr_id):
    type_str, id_str = in_predr_id.split('_')
    if "refnr" in type_str:
        n_fheader = refine_snode_fname_header
        s_fheader = pred_rif_fname_header
    else:
        n_fheader = train_snode_fname_header
        s_fheader = pred_trn_fname_header
    snod_fname = n_fheader+id_str+'_'+in_key
    output_fname = s_fheader+id_str+'_'+in_key+'_'
    return snod_fname, output_fname



class Data_Definder(object):
    def __init__(self):
        super(Data_Definder, self).__init__()
        #-*************************
        seqs_dim = 33
        # Typ1=many meaningful 0 and all>=0
        # Typ2=random around 0, possible <0 and >0 or only >0
        # Typ3=after offset become Typ1
        # Typ4=after offset become Typ2
        # 0_Month 1_Day 2_Year 3_PRECIPmm(Typ1) 4_TMP_MXdgC(Typ2) 5_TMP_MNdgC(Typ2) 6_SOLARMJ.m2(Typ2) 7_WINDM.s(Typ2) 8_RHmd(Typ2) 9_SNOFALLmm(Typ1)
        # 10_SNOMELTmm(Typ1) 11_SNOmm(Typ1) 12_PETmm(Typ1) 13_ETmm(Typ1) 14_SW_INITmm(Typ4,-25) 15_SW_ENDmm(Typ4,-25) 16_PERCmm(Typ1) 17_GW_RCHGmm(Typ1)
        # 18_DA_RCHGmm(Typ1) 19_REVAPmm(Typ1) 20_SA_STmm(Typ3,-750) 21_DA_STmm(Typ3,-2000) 22_SURQ_GENmm(Typ1) 23_SURQ_CNTmm(Typ1) 24_TLOSSmm(Typ1,all0) 25_LATQGENmm(Typ1)
        # 26_LATQCNTmm(Typ1) 27_GW_Qmm(Typ1,all0) 28_GW_Q_Dmm(Typ1)  29_DAILYCN(Typ4,-70) 30_LAI(Typ1) 31_QTILEmm(Typ1,all0) 32_Q_pred_mm(Typ1)
        data_offsets = [0 for i in range(seqs_dim)]
        data_offsets[14] = 25
        data_offsets[15] = 25
        data_offsets[20] = 750
        data_offsets[21] = 2000
        data_offsets[29] = 70
        
        times_idx = [3, 9, 10, 11, 16, 17, 18, 22, 23, 32]
        data_times = np.array([1 for i in range(seqs_dim)])
        data_times[times_idx] = 5
        
        #-***************************
        month_names = ['Jan','Feb',"Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        cond_names = ['Month', 'Day', 'WINDM', 'RHmd', 'TMP', 'SOLAR', 'PRECIP']
        cond_cols = [[0], [1], [7], [8], [4,5], [6], [3]]
        varb_names = cond_names
        varb_names.extend(['ET', 'SNO', 'SW', 'SA', 'SURQ', 'LATQ', 'GW', 'Q_pred'])
        varb_cols = cond_cols
        varb_cols.extend([[12,13], [9,10,11], [14,15], [18,19,20,21], [22,23,29,30], [16,25,26,31], [17,27,28], [32]])
        node_names = ["N0_a_C1", "N1_b_C2", "N2_c_Eva", "N3_d_SP", "N4_e_SW", 
                           "N5_f_Aq", "N6_g_SR", "N7_h_La", "N8_i_BF", "N9_j_Q"]
        node_consist = {
                "a": [['Month'], ['WINDM','RHmd','TMP','SOLAR'], True],
                "b": [['Month'], ['TMP','SOLAR','PRECIP'], True],
                "c": [['Month'], ['ET'], False],
                "d": [['Month'], ['SNO'], False],
                "e": [['Month'], ['SW'], False],
                "f": [['Month'], ['SA'], False],
                "g": [['Month'], ['SURQ'], False],
                "h": [['Month'], ['LATQ'], False],
                "i": [['Month'], ['GW'], False],
                "j": [['Month'], ['Q_pred'], False]
                }
                     # a, b, c, d, e, f, g, h, i, j
        adj_matrix = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # N0_a_C1
                      [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # N1_b_C2
                      [0, 0, 0, 1, 1, 0, 1, 0, 0, 0],  # N2_c_Eva
                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],  # N3_d_SP
                      [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],  # N4_e_SW
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # N5_f_Aq
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # N6_g_SR
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # N7_h_La
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # N8_i_BF
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # N9_j_Q
                     # a, b, c, d, e, f, g, h, i, j
        steps_matrix = [[0, 0, 5, 0, 0, 0, 0, 0, 0, 0],  # N0_a_C1
                       [0, 0, 0, 5, 5, 0, 0, 0, 0, 0],  # N1_b_C2
                       [0, 0, 0, 10, 10, 0, 10, 0, 0, 0],  # N2_c_Eva
                       [0, 0, 0, 0, 0, 0, 10, 10, 10, 0],  # N3_d_SP
                       [0, 0, 0, 0, 0, 10, 10, 10, 0, 0],  # N4_e_SW
                       [0, 0, 0, 0, 0, 0, 0, 0, 10, 0],  # N5_f_Aq
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],  # N6_g_SR
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 10],  # N7_h_La
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 10],  # N8_i_BF
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # N9_j_Q
                     # a, b, c, d, e, f, g, h, i, j
        det_matrix = [[0, 0, 1, 1, 1, 1, 0, 0, 0, 0],  # N0_a_C1
                      [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],  # N1_b_C2
                      [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],  # N2_c_Eva
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],  # N3_d_SP
                      [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # N4_e_SW
                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],  # N5_f_Aq
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # N6_g_SR
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # N7_h_La
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # N8_i_BF
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # N9_j_Q
        self.varb_col_names = varb_names
        self.varb_col_idxs = varb_cols
        self.node_names = node_names
        self.node_consist = node_consist
        self.month_names = month_names
        self.adj_matrix = np.array(adj_matrix)
        self.steps_matrix = np.array(steps_matrix)
        self.detect_matrix = np.array(det_matrix)
        self.node_keys = list(node_consist.keys())
        self.seqs_dim = seqs_dim
        self.data_offsets = data_offsets
        self.data_times = data_times
        self.data_zeror = np.zeros(seqs_dim)
    def __call__(self, in_dataset, in_scaler):
        data_scl = in_dataset - self.data_offsets
        data_scl.dtype = 'float32'
        data_scl[:,3:] = in_scaler.fit_transform(data_scl)[:,3:]
        data_scl = data_scl/self.data_times
        month_vec = data_scl[:,0]
        month_mat = self.dummy_month(month_vec)
        data_scl = np.concatenate((month_mat,data_scl[:,1:]), axis=1)
        data_scl.dtype = in_dataset.dtype
        self.scaler = in_scaler
        return data_scl
    def recover(self, in_dataset):
        month_rcv = self.press_month(in_dataset[:,:12])
        month_rcv = tf.cast(month_rcv, dtype=in_dataset.dtype)
        month_rcv = tf.reshape(month_rcv,[-1,1])
        data_rcv = tf.concat((month_rcv, in_dataset[:,12:]), axis=1)
        data_rcv = data_rcv * self.data_times
        if not hasattr(data_rcv, "numpy"):
            return data_rcv
        if  tf.executing_eagerly():
            data_rcv = data_rcv.numpy()
        else:
            data_rcv = data_rcv.eval()
        data_rcv = np.array(data_rcv, dtype='float32')
        data_rcv[:,3:] = self.scaler.inverse_transform(data_rcv)[:,3:]
        data_rcv = data_rcv + self.data_offsets
        data_rcv = tf.cast(data_rcv, dtype=in_dataset.dtype)
        return data_rcv
    def dummy_month(self, in_vec):
        month_dummy = pd.get_dummies(in_vec).values.tolist()
        return month_dummy
    def press_month(self, in_mat):
        w_mat = np.zeros([in_mat.shape[0], 0])
        for j in range(in_mat.shape[1]):
            w_vec = np.ones([in_mat.shape[0], 1]) * j
            w_mat = np.concatenate([w_mat, w_vec], 1)
        w_mat = tf.convert_to_tensor(w_mat, dtype = in_mat.dtype)
        res_mat = tf.math.multiply(in_mat, w_mat)
        res_vec = tf.math.reduce_sum(res_mat, 1)
        return res_vec
    def get_date_cols(self):
        return [0,1,2]
    def get_cols(self, in_keys, scl_tag=True):
        cond_csst_names = []
        attr_csst_names = []
        for k in in_keys:
            n_consist = self.node_consist[k]
            cond_csst_names.extend(n_consist[0])
            attr_csst_names.extend(n_consist[1])
        cond_csst_names = list(np.unique(cond_csst_names))
        attr_csst_names = list(np.unique(attr_csst_names))
        cond_len = len(cond_csst_names)
        attr_len = len(attr_csst_names)
        out_c_cols = []; out_c_names = [];
        out_x_cols = []; out_x_names = [];
        for c in range(cond_len+attr_len):
            if c < cond_len:
                name = cond_csst_names[c]
            else:
                name = attr_csst_names[c-cond_len]
            name_idx = self.varb_col_names.index(name)
            cols = self.varb_col_idxs[name_idx]
            if len(cols)>1:
                name_list = [name+'_'+str(i) for i in range(len(cols))]
            else:
                name_list = [name]
            if c < cond_len:
                out_c_cols.extend(cols)
                out_c_names.extend(name_list)
            else:
                out_x_cols.extend(cols)
                out_x_names.extend(name_list)
        if scl_tag:
            out_c_cols = list(np.array(out_c_cols) + 11)
            out_x_cols = list(np.array(out_x_cols) + 11)
            if "Month" in out_c_names:
                m_idx = out_c_names.index("Month")
                out_c_names = out_c_names[:m_idx] + self.month_names + out_c_names[(m_idx+1):]
                out_c_cols = out_c_cols[:m_idx] + [i for i in range(12)] + out_c_cols[(m_idx+1):]
            if "Month" in out_x_names:
                m_idx = out_x_names.index("Month")
                out_x_names = out_x_names[:m_idx] + self.month_names + out_x_names[(m_idx+1):]
                out_x_cols = out_x_cols[:m_idx] + [i for i in range(12)] + out_x_cols[(m_idx+1):]
        return [out_x_cols, out_x_names], [out_c_cols, out_c_names]


class Data_Store(object):
    def __init__(self):
        super(Data_Store, self).__init__()
        self.fname = data_source_fname
        self.split_rate = 1
        self.d_df = Data_Definder()
        self.node_consist = self.d_df.node_consist # by default
        self.node_keys = self.d_df.node_keys # by default
        self.node_names = self.d_df.node_names # by default
        self.node_num = len(self.node_keys)
        self.adj_matrix = self.d_df.adj_matrix # by default
        mat = self.adj_matrix
        self.detect_matrix = self.d_df.detect_matrix
        from_arr = np.sum(mat,0)
        to_arr = np.sum(mat,1)
        sources = [k for i,k in enumerate(self.node_keys) if from_arr[i]==0]
        targets = [k for i,k in enumerate(self.node_keys) if to_arr[i]==0]
        self.node_sources = sources
        self.node_target = targets[-1]
        self.mask_th_dict={'a':0.0005, 'b':0.0005, 'c':0.0005, 'd':0.0005,
                 'e':0.5, 'f':0.0005, 'g':0.00005, 'h':0.0005, 'i':0.00005, 'j':0.00005}
        return
    def get_mask_th(self, in_keys):
        if isinstance(in_keys, str):
            in_keys = [char for char in in_keys]
        th_arr = []
        for k in in_keys:
            th_arr.append(self.mask_th_dict[k])
        return np.min(th_arr)
    def cmb_sources(self, in_keys):
        if isinstance(in_keys, str):
            in_keys = [char for char in in_keys]
        source_tags = [k in self.node_sources for k in in_keys]
        if any(source_tags):
            n_sources = len(self.node_sources)
            s_loc = source_tags.index(True)
            in_keys.pop(s_loc)
            for i in range(n_sources-1, -1, -1):
                in_keys.insert(s_loc, self.node_sources[i])
        in_keys = list(np.unique(in_keys))
        return in_keys
    def get_cmb_cols(self, in_keys, scl_tag=True):
        if isinstance(in_keys, str):
            in_keys = [char for char in in_keys]
        the_keys = in_keys.copy()
        the_keys = self.cmb_sources(the_keys)
        return self.d_df.get_cols(the_keys, scl_tag)
    def load(self):
        if exists(data_obj_fname):
            print("[Data_Store] loading saved data.")
            saved_data = load_obj(data_obj_fname)
            self.data_resource = saved_data[0]
            self.data_scl = saved_data[1]
            self.scaler = saved_data[2]
            self.d_df.scaler = self.scaler
        else:
            print("[Data_Store] reading ", self.fname, " ...")
            self.data_resource = pd.read_csv(self.fname,delimiter=",",skiprows=1).values
            data_scl = self.data_resource.copy()
            scaler = StandardScaler(with_mean=False, copy=True)
            self.data_scl = self.d_df(data_scl, scaler)
            self.scaler = self.d_df.scaler
            self.save()
        seqs_len = self.data_resource.shape[0]
        split_point = int(np.floor(seqs_len*self.split_rate))
        self.train_seqs = range(split_point)
        self.test_seqs = range(split_point,seqs_len)
        return
    def save(self):
        saved_data = list([self.data_resource, self.data_scl, self.scaler])
        save_obj(saved_data, data_obj_fname)
        return
    def __call__(self, in_keys, scl_tag):
        if isinstance(in_keys, str):
            in_keys = [char for char in in_keys]
        if not all([k in self.node_keys for k in in_keys]):
            raise IndexError("[Data_Store] index out of range")
        #------
        if scl_tag:
            call_data = self.data_scl
        else:
            call_data = self.data_resource
        [attr_cols, attr_names], [cond_cols, cond_names] = self.get_cmb_cols(in_keys, scl_tag)
        cond_data = call_data[:,cond_cols]
        attr_data = call_data[:,attr_cols]
        if len(cond_cols)>0:
            data_cond_tr = cond_data[self.train_seqs,:]
            data_cond_ts = cond_data[self.test_seqs,:]
        else:
            data_cond_tr = data_cond_ts = []
        if len(attr_cols)>0:  
            data_attr_tr = attr_data[self.train_seqs,:]
            data_attr_ts = attr_data[self.test_seqs,:]
        else:
            data_attr_tr = data_attr_ts = []
        #------
        data_tr_list = [data_attr_tr, data_cond_tr]
        data_ts_list = [data_attr_ts, data_cond_ts]
        names_list = [attr_names, cond_names]
        return data_tr_list, data_ts_list, names_list
    def get_node_name(self, node_key):
        node_idx = self.node_keys.index(node_key)
        return self.node_names[node_idx]
    def get_names(self, in_keys):
        [attr_cols, attr_names], [cond_cols, cond_names] = self.get_cmb_cols(in_keys)
        return [attr_names, cond_names]
    def get_unit_step(self, key_A, key_B):
        A_idx = self.node_keys.index(key_A)
        B_idx = self.node_keys.index(key_B)
        n_step = self.d_df.steps_matrix[A_idx, B_idx]
        return n_step
    def get_merged_step(self, causes_keys, result_key):
        causes_steps = []
        for c in causes_keys:
            causes_steps.append(self.get_unit_step(c, result_key))
        n_step = np.max(causes_steps)
        return n_step
    def get_dims(self, in_keys):
        [attr_cols, attr_names], [cond_cols, cond_names] = self.get_cmb_cols(in_keys)
        x_dim = len(attr_names)
        c_dim = len(cond_names)
        return [x_dim, c_dim]
    def get_seeds(self, in_keys, in_batch_size, batch_idx=0):
        [attr_cols, attr_names], [cond_cols, cond_names] = self.get_cmb_cols(in_keys)
        batch_range = range(batch_idx*in_batch_size, (batch_idx+1)*in_batch_size)
        cond_seed = self.data_scl[batch_range][:,cond_cols]
        attr_seed = self.data_scl[batch_range][:,attr_cols]
        cond_seed = tf.convert_to_tensor(cond_seed, dtype='float32')
        attr_seed = tf.convert_to_tensor(attr_seed, dtype='float32')
        return attr_seed, cond_seed
    def get_train(self, in_keys, scl_tag=True):
        data_tr_list, _, _ = self.__call__(in_keys, scl_tag)
        x_train = data_tr_list[0]
        c_train = data_tr_list[1]
        x_train = tf.convert_to_tensor(x_train, dtype='float32')
        c_train = tf.convert_to_tensor(c_train, dtype='float32')
        return x_train, c_train
    def get_test(self, in_keys, scl_tag=True):
        _, data_ts_list, _ = self.__call__(in_keys, scl_tag)
        x_test = data_ts_list[0]
        c_test = data_ts_list[1]
        x_test = tf.convert_to_tensor(x_test, dtype='float32')
        c_test = tf.convert_to_tensor(c_test, dtype='float32')
        return x_test, c_test
    # ---
    def get_seeds_bundle(self, in_keys, in_batch_size, batch_idx=0):
        if isinstance(in_keys, str):
            in_keys = [char for char in in_keys]
        x_seed_bundle = []
        c_seed_bundle = []
        for k in in_keys:
            x_seed, c_seed = self.get_seeds(k, in_batch_size, batch_idx)
            x_seed_bundle.append(x_seed)
            c_seed_bundle.append(c_seed)
        return x_seed_bundle, c_seed_bundle
    def get_train_bundle(self, in_keys, scl_tag=True):
        if isinstance(in_keys, str):
            in_keys = [char for char in in_keys]
        x_bundle = []
        c_bundle = []
        for k in in_keys:
            x_train, c_train = self.get_train(k, scl_tag)
            x_bundle.append(x_train)
            c_bundle.append(c_train)
        return x_bundle, c_bundle
    def get_test_bundle(self, in_keys, scl_tag=True):
        if isinstance(in_keys, str):
            in_keys = [char for char in in_keys]
        x_bundle = []
        c_bundle = []
        for k in in_keys:
            x_test, c_test = self.get_test(k, scl_tag)
            x_bundle.append(x_test)
            c_bundle.append(c_test)
        return x_bundle, c_bundle
    # ---
    def get_x_train(self, in_keys, scl_tag=True):
        return self.get_train(in_keys, scl_tag)[0]
    def get_c_train(self, in_keys, scl_tag=True):
        return self.get_train(in_keys, scl_tag)[1]
    def get_x_test(self, in_keys, scl_tag=True):
        return self.get_test(in_keys, scl_tag)[0]
    def get_c_test(self, in_keys, scl_tag=True):
        return self.get_test(in_keys, scl_tag)[1]
    def get_batch(self, in_keys, in_step, in_batch_size, train_tag, scl_tag):
        if train_tag:
            idxs_seqs = self.train_seqs
            x_full, c_full = self.get_train(in_keys, scl_tag)
        else:
            idxs_seqs = self.test_seqs
            x_full, c_full = self.get_test(in_keys, scl_tag)
        bt_start = in_step*in_batch_size
        bt_end = (in_step+1)*in_batch_size
        if bt_end > len(idxs_seqs):
            bt_end = len(idxs_seqs)
        x_batch = x_full[bt_start:bt_end,:]
        c_batch = c_full[bt_start:bt_end,:]
        return x_batch, c_batch
    # ---
    def get_batch_orig(self, in_keys, in_step, in_batch_size, train_tag=True):
        orig_x_batch, orig_c_batch = self.get_batch(in_keys, in_step, in_batch_size, train_tag, False)
        return orig_x_batch, orig_c_batch
    def get_batch_recv(self, in_keys, in_step, in_batch_size, train_tag=True):
        recv_x_batch, recv_c_batch = self.get_batch(in_keys, in_step, in_batch_size, train_tag, True)
        return recv_x_batch, recv_c_batch
    def get_batch_bundle(self, in_keys, in_step, in_batch_size, train_tag, scl_tag):
        if train_tag:
            idxs_seqs = self.train_seqs
            get_full_fun = self.get_train
        else:
            idxs_seqs = self.test_seqs
            get_full_fun = self.get_test
        bt_start = in_step*in_batch_size
        bt_end = (in_step+1)*in_batch_size
        if bt_end > len(idxs_seqs):
            bt_end = len(idxs_seqs)
        x_batch_bundle = []
        c_batch_bundle = []
        for k in in_keys:
            x_full, c_full = get_full_fun(k, scl_tag)
            x_batch = x_full[bt_start:bt_end,:]
            c_batch = c_full[bt_start:bt_end,:]
            x_batch_bundle.append(x_batch)
            c_batch_bundle.append(c_batch)
        return x_batch_bundle, c_batch_bundle
    def get_batch_bundle_orig(self, in_keys, in_step, in_batch_size, train_tag=True):
        orig_x_batch_bundle, orig_c_batch_bundle = self.get_batch_bundle(in_keys, in_step, in_batch_size, train_tag, False)
        return orig_x_batch_bundle, orig_c_batch_bundle
    def get_batch_bundle_recv(self, in_keys, in_step, in_batch_size, train_tag=True):
        recv_x_batch_bundle, recv_c_batch_bundle = self.get_batch_bundle(in_keys, in_step, in_batch_size, train_tag, True)
        return recv_x_batch_bundle, recv_c_batch_bundle
    # ---
    def get_dates(self, in_step, in_batch_size, train_tag=True):
        MDY_cols = self.d_df.get_date_cols() # 'Month' 'Day' 'Year'
        if train_tag:
            idxs_seqs = self.train_seqs
        else:
            idxs_seqs = self.test_seqs
        bt_start = in_step*in_batch_size
        bt_end = (in_step+1)*in_batch_size
        if bt_end > len(idxs_seqs):
            bt_end = len(idxs_seqs)
        bt_seqs = idxs_seqs[bt_start:bt_end]
        date_vecs = self.data_resource[bt_seqs,:][:,MDY_cols]
        return date_vecs
    def descale_x(self, in_keys, in_x_data):
        if isinstance(in_keys, str):
            in_keys = [char for char in in_keys]
        if not all([k in self.node_keys for k in in_keys]):
            raise IndexError("[Data_Store] index out of range")
        #------
        [attr_cols, _], _ = self.get_cmb_cols(in_keys, scl_tag=True)
        data_template_scl = tf.convert_to_tensor(np.zeros([in_x_data.shape[0], 1]), dtype='float32')
        for j in range(1, self.data_scl.shape[1]):
            if not j in attr_cols:
                j_col = np.zeros([in_x_data.shape[0], 1])
            else:
                j_idx = attr_cols.index(j)
                j_col = in_x_data[:,j_idx:(j_idx+1)]
            j_col = tf.convert_to_tensor(j_col, dtype='float32')
            data_template_scl = tf.concat([data_template_scl, j_col], axis=1)
        if not hasattr(data_template_scl, "numpy"):
            return in_x_data
        data_template_bk = self.d_df.recover(data_template_scl)
        [attr_cols_bk, _], _ = self.get_cmb_cols(in_keys, scl_tag=False)
        data_bk = np.array(data_template_bk.numpy())[:,np.array(attr_cols_bk)]
        return data_bk



class Graph_Obj(object):
    def __init__(self, in_data_obj):
        super(Graph_Obj, self).__init__()
        print("[Graph_Obj] creating...")
        self.adj_mat = np.array(in_data_obj.adj_matrix)
        self.det_mat = np.array(in_data_obj.detect_matrix)
        self.edge_num = np.sum(self.adj_mat)
        self.node_keys = in_data_obj.node_keys # list
        self.node_names = in_data_obj.node_names
        self.node_num = len(self.node_keys)
        out_counts = np.zeros(self.node_num,)
        in_counts = np.zeros(self.node_num,)
        node_depths = np.zeros(self.node_num,)
        for i in range(self.node_num):
            for j in range(self.node_num):
                if self.adj_mat[i,j]>0:
                    out_counts[i] += 1
                    in_counts[j] += 1
                    node_depths[j] = node_depths[i] + 1
        self.out_counts = out_counts
        self.in_counts = in_counts
        self.node_depths = node_depths
        self.depth = max(node_depths)
    
    def layer(self, in_depth):
        return self.node_depths[self.node_depths==in_depth]
    def get_name(self, node_key):
        node_idx = self.node_keys.index(node_key)
        return self.node_names[node_idx]
    def path(self, in_node, out_node): # recursive
        if in_node == out_node:
            return [out_node]
        in_node_idx = self.node_keys.index(in_node)
        child_nodes = np.array(self.adj_mat[in_node_idx,:])
        child_nodes = np.array([i for i in range(len(child_nodes)) if child_nodes[i]>0])
        out_path = []
        for ch in child_nodes:
            out_path = self.path(self.node_keys[ch], out_node)
            if len(out_path)>0:
                out_path.insert(0,in_node)
                break
        return out_path
    def paths(self, in_node, out_node): # recursive
        if in_node == out_node:
            return [[out_node]]
        in_node_idx = self.node_keys.index(in_node)
        child_nodes = np.array(self.adj_mat[in_node_idx,:])
        child_nodes = np.array([i for i in range(len(child_nodes)) if child_nodes[i]>0])
        out_paths = []
        for ch in child_nodes:
            my_paths = self.paths(self.node_keys[ch], out_node)
            if len(my_paths)>0:
                for p in my_paths:
                    p.insert(0,in_node)
                out_paths.extend(my_paths)
        return out_paths
    def check_reach(self, in_nodes, out_node, in_slt_edges=[]):
        if len(in_slt_edges)>0:
            orig_adj_mat = self.adj_mat.copy()
            tmp_adj_mat = np.zeros_like(orig_adj_mat)
            for e in in_slt_edges:
                e0 = self.node_keys.index(e[0])
                e1 = self.node_keys.index(e[1])
                tmp_adj_mat[e0, e1] = 1
            self.adj_mat = tmp_adj_mat
        paths_arr = []
        rch_tags = []
        for n in in_nodes:
            my_paths = self.paths(n, out_node)
            if len(my_paths)>0:
                rch_tags.append(True)
            else:
                rch_tags.append(False)
            for p in my_paths:
                for i in range(1,len(p)):
                    e0 = p[i-1]
                    e1 = p[i]
                    if not [e0,e1] in paths_arr:
                        paths_arr.append([e0,e1])
        rch_edges = []
        for e in in_slt_edges:
            if e in paths_arr:
                rch_edges.append(e)
        if len(in_slt_edges)>0:
            self.adj_mat = orig_adj_mat
        return rch_tags, rch_edges
        
    def nodes_out(self, in_node):
        node_idx = self.node_keys.index(in_node)
        keys_arr = np.array(self.node_keys)
        adj_arr = np.array(self.adj_mat[node_idx,:])
        child_nodes = keys_arr[adj_arr>0]
        return child_nodes
    def nodes_detect_out(self, in_node):
        node_idx = self.node_keys.index(in_node)
        keys_arr = np.array(self.node_keys)
        det_arr = np.array(self.det_mat[node_idx,:])
        det_ch_nodes = keys_arr[det_arr>0]
        return det_ch_nodes
    
    def nodes_in(self, in_node):
        node_idx = self.node_keys.index(in_node)
        keys_arr = np.array(self.node_keys)
        adj_arr = np.array(self.adj_mat[:,node_idx])
        parent_nodes = keys_arr[adj_arr>0]
        return parent_nodes
    def nodes_detect_in(self, in_node):
        node_idx = self.node_keys.index(in_node)
        keys_arr = np.array(self.node_keys)
        det_arr = np.array(self.det_mat[:,node_idx])
        det_pr_nodes = keys_arr[det_arr>0]
        return det_pr_nodes
    
    def update_edge(self, inputs_lists):
        edge = inputs_lists[0]
        value = inputs_lists[1]
        self.adj_mat[edge[0],edge[1]] = value
        return self.adj_mat
    
    def set_order(self, node_arr, in_depth):
        self.layer_orders[in_depth] = node_arr
        return

# ---- for general graph ----
def check_reach(in_edges, in_nodes, out_node): # unused
    all_nodes = []
    for e in in_edges:
        all_nodes.extend(e)
    all_nodes = list(np.unique(all_nodes))
    all_nodes.sort()
    adj_dict = {n:[0 for m in all_nodes] for n in all_nodes}
    for e in in_edges:
        adj_dict[e[1]][all_nodes.index(e[0])] = 1
    rch_nodes = [] # reversely reached
    check_nodes = []
    check_nodes.extend(out_node)
    while len(check_nodes)>0:
        n = check_nodes.pop(0)
        n_parents = [all_nodes[i] for i,j in enumerate(adj_dict[n]) if j==1]
        n_parents = [n for n in n_parents if not n in rch_nodes]
        check_nodes.extend(n_parents)
        rch_nodes.extend(n_parents)
    rch_tags = [n in rch_nodes for n in in_nodes]
    return rch_tags

def edges_to_nodes(in_edges):
    # --- fully reached order nodes ---
    rch_nodes = [e[1] for e in in_edges]
    rch_nodes_idxs = sorted(np.unique(rch_nodes, return_index=True)[1])
    rch_nodes = [rch_nodes[i] for i in rch_nodes_idxs] # in reached order
    all_nodes = []
    for e in in_edges:
        all_nodes.extend(e)
    all_nodes = list(np.unique(all_nodes))
    parents_dict = {n:[e[0] for e in in_edges if e[1]==n] for n in rch_nodes}
    full_rch_nodes = [] # in fully reached order
    for e in in_edges:
        pts = parents_dict[e[1]]
        _ = pts.pop(pts.index(e[0]))
        if len(pts)==0: # fully reached
            full_rch_nodes.append(e[1])
    return full_rch_nodes
    
def depth_to_breadth(depth_ordered_edges, in_nodes):
    all_nodes = []
    for e in depth_ordered_edges:
        all_nodes.extend(e)
    all_nodes = list(np.unique(all_nodes))
    source_nodes = [n for n in in_nodes if n in all_nodes]
    rch_nodes = [e[1] for e in depth_ordered_edges]
    rch_nodes_idxs = sorted(np.unique(rch_nodes, return_index=True)[1])
    rch_nodes = [rch_nodes[i] for i in rch_nodes_idxs] # in reached order
    # --- to remove specified source nodes ---
    rch_nodes = [n for n in rch_nodes if not n in source_nodes]
    # --- to remove unreached/updtream nodes ---
    rch_nodes = [n for n in rch_nodes if any(check_reach(depth_ordered_edges, source_nodes, n))]
    saw_nodes = np.unique(np.concatenate([source_nodes, rch_nodes]))
    saw_nodes.sort()
    saw_edges = [e for e in depth_ordered_edges if (e[0] in saw_nodes) and (e[1] in saw_nodes)]
    # --- to find breadth first order edges (= fully reached ordered nodes) ---
    buildup_edges = [] # breadth first order
    parents_dict = {n:[e[0] for e in saw_edges if e[1]==n] for n in rch_nodes}
    full_rch_nodes = source_nodes.copy()
    stop_tag = False
    while not stop_tag:
        for n in rch_nodes:
            if len(parents_dict[n])==0:
                continue
            pts = parents_dict[n]
            n_rch_edges = [[p,n] for p in pts if p in full_rch_nodes]
            parents_dict[n] = [p for p in pts if not p in full_rch_nodes]
            buildup_edges.extend(n_rch_edges)
            if len(parents_dict[n])==0: # n is fully reached
                full_rch_nodes.append(n)
                break
        stop_tag = len(full_rch_nodes)==len(saw_nodes) and all(np.sort(full_rch_nodes)==saw_nodes)
    return buildup_edges, full_rch_nodes
    



#    # check txt
#    f=open(fname, 'r', encoding="utf-8").readlines()
#    N=len(f)-1
#    for i in range(0,N):
#        w=f[i].split()
#        l1=w[1:8]
#        l2=w[8:15]
#        try:
#            list1=[float(x) for x in l1]
#            list2=[float(x) for x in l2]
#        except ValueError:
#            print("error on line",i)
#        result=stats.ttest_ind(list1,list2)
#        print(result[1])







    
