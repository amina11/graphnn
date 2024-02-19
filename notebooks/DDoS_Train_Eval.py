### this is to train the graphnn
import sys
sys.executable

import os
import sys
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import ogb
from tqdm import tqdm
import hiplot as hip
from copy import deepcopy
import datetime
import json
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset, TensorDataset
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw


from torch_geometric.data import Data
cwd = os.getcwd()
print(cwd)
cwd_parent = os.path.abspath(os.path.join(cwd, os.pardir))
print(cwd_parent)

sys.path.append(cwd_parent)

import deepadr
from deepadr.dataset import *
from deepadr.utilities import *
from deepadr.chemfeatures import *
from deepadr.train_functions import *
from deepadr.model_gnn_ogb import GNN, DeepAdr_SiameseTrf, ExpressionNN

partition_style ='Drug1_ID' #'Drug2_ID' # 'Cell_Line_ID'
rawdata_dir = '../data/raw/'
processed_dir = '../data/processed/'
up_dir = '..'

report_available_cuda_devices()
n_gpu = torch.cuda.device_count()
n_gpu
device_cpu = get_device(to_gpu=True)
# device_gpu = get_device(True, index=0)
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)

score = 'zip_thresh' #'hsa_thresh' #'loewe_thresh'
score_val = 1

DSdataset_name = f'DrugComb_{score}_{score_val}'
data_fname = 'data_v1' # v2 for baseline models, v3 for additive samples
targetdata_dir = create_directory(os.path.join(processed_dir, DSdataset_name, data_fname))
targetdata_dir_raw = create_directory(os.path.join(targetdata_dir, "raw"))
targetdata_dir_processed = create_directory(os.path.join(targetdata_dir, "processed"))
targetdata_dir_exp = create_directory(os.path.join(targetdata_dir, "experiments"))
# # ReaderWriter.dump_data(dpartitions, os.path.join(targetdata_dir, 'data_partitions.pkl'))
print(targetdata_dir)


# Make sure to first run the "DDoS_Dataset_Generation" notebook first
dataset = MoleculeDataset(root=targetdata_dir)



import pickle
with open(targetdata_dir_raw +'/data_pairs.pkl', 'rb') as file:
    data_pairs = pickle.load(file )
    
    
## This file is to leav a cell out partition
def   data_partition_leave_cells_out(num_folds,random_state,valid_set_portion,path_to_save, feature ):
    import pickle
    with open(targetdata_dir_raw +'/data_pairs.pkl', 'rb') as file:
        data_pairs = pickle.load(file )
        
    #grouped = data_pairs.groupby('Cosmic_ID')['Y'].value_counts(normalize=True).unstack(fill_value=0)
    grouped = data_pairs.groupby(feature)['Y'].value_counts(normalize=True).unstack(fill_value=0)
    ### we lable those have mostly 0 as class0, blanced 0 and 1 as class 1 and similar distribution to 
    ## the original class distribution, around 0.81 as 2 and more balanced ones are 3
    label = []
    for i in range(len(grouped)):
        if grouped[0].iloc[i]>=0.90:
            label.append(0)
        elif grouped[0].iloc[i]<0.2:
            label.append(1)
        elif grouped[0].iloc[i]<0.90 and grouped[0].iloc[i]>0.75:
            label.append(2)
        else:
            label.append(3)   
            
    grouped['label'] = label
    grouped = grouped.reset_index()
    
    X = grouped[feature]
    Y = grouped['label']
    
    skf_trte = StratifiedKFold(n_splits=num_folds, random_state=random_state, shuffle=True)  # split train and test
    skf_trv = StratifiedShuffleSplit(n_splits=2, 
                                     test_size=valid_set_portion, 
                                     random_state=random_state)  # split train and test

    data_partition = {}
    fold_num = 0
    
    for train_index, test_index in skf_trte.split(X,Y):
        x_tr = np.zeros(len(train_index))
        y_tr = Y[train_index] 

        for tr_index, val_index in skf_trv.split(x_tr, y_tr):
            tr_ids = train_index[tr_index]
            val_ids = train_index[val_index]
            data_partition[fold_num] = {'train': tr_ids,
                                             'validation': val_ids,
                                             'test': test_index}
        fold_num = fold_num + 1
    
    
    data_partitions = {}
   
    YY = data_pairs.Y
    for i in range(num_folds):
        train_cell = X[data_partition[i]['train']]
        val_cell = X[data_partition[i]['validation']]
        test_cell = X[data_partition[i]['test']]
        train_indices = np.array(data_pairs.loc[data_pairs[feature].isin(train_cell)].index)
        val_indices = np.array(data_pairs.loc[data_pairs[feature].isin(val_cell)].index)
        test_indices = np.array(data_pairs.loc[data_pairs[feature].isin(test_cell)].index)
        data_partitions[i] = {'train': train_indices,
                                             'validation': val_indices,
                                             'test': test_indices}
        
    
    
        print("fold_num:", i)
        print('train data')
        report_label_distrib(YY[data_partitions[i]['train']])
        print('validation data')
        report_label_distrib(YY[data_partitions[i]['validation']])
        print('test data')
        report_label_distrib(YY[data_partitions[i]['test']])
        print()
        fold_num += 1
        print("-"*25)
    
    file_name = path_to_save + f'/partions_{feature}_out.pkl'
    with open(file_name, "wb") as pickle_file:
        pickle.dump(data_partitions, pickle_file)
    return data_partitions


def report_label_distrib(labels):
    classes, counts = np.unique(labels, return_counts=True)
    norm_counts = counts/counts.sum()
    for i, label in enumerate(classes):
        print("class:", label, "norm count:", norm_counts[i])


def validate_partitioan(train_indices, val_indices, test_indices):
    # Find the intersection between test_indices and val_indices
    test_val_intersection = set(test_indices).intersection(val_indices)

    # Find the intersection between test_indices and train_indices
    test_train_intersection = set(test_indices).intersection(train_indices)

    # Find the intersection between val_indices and train_indices
    val_train_intersection = set(val_indices).intersection(train_indices)

    # Check if any of the intersections have common elements
    if test_val_intersection or test_train_intersection or val_train_intersection:
        print("The sets of indices intersect.")
    else:
        print("The sets of indices do not intersect.")

    # Calculate the total count of indices
    total_indices_count = len(test_indices) + len(val_indices) + len(train_indices)

    # Print the total count of indices
    print("Total count of indices:", total_indices_count)
    
    
    
data_partitions =data_partition_leave_cells_out(5, 42, 0.1, targetdata_dir_processed,partition_style )



fold_partitions = data_partitions

# training parameters

tp = {
    "batch_size" : 300,
    "num_epochs" : 100,
    
    "emb_dim" : 100,
    "gnn_type" : "gatv2",
    "num_layer" : 5,
    "graph_pooling" : "mean", #attention
    
    "input_embed_dim" : None,
    "gene_embed_dim": 1,
    "num_attn_heads" : 2,
    "num_transformer_units" : 1,
    "p_dropout" : 0.3,
    "nonlin_func" : 'nn.ReLU()',
    "mlp_embed_factor" : 2,
    "pooling_mode" : 'attn',
    "dist_opt" : 'cosine',

    "base_lr" : 3e-4, #3e-4
    "max_lr_mul": 10,
    "l2_reg" : 1e-7,
    "loss_w" : 1.,
    "margin_v" : 1.,

    "expression_dim" : 64,
    "expression_input_size" : 908,
    "exp_H1" : 4096,
    "exp_H2" : 1024
}

partition = fold_partitions
gpu_num = 1
time_stamp = datetime.datetime.now().strftime('%Y-%m-%d')
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
queue = mp.Queue()

for i in range(len(fold_partitions)):
    partition = fold_partitions[i]
    exp_dir = create_directory(os.path.join(targetdata_dir_exp, str(partition_style)+"_fold_"+str(i)+"_"+time_stamp))
    create_directory(os.path.join(exp_dir, "predictions"))
    create_directory(os.path.join(exp_dir, "modelstates"))
    deepadr.train_functions.run_exp(queue, dataset, 0, tp, exp_dir, partition)
print("End: " + datetime.datetime.now().strftime('%Y-%m-%d')) 




