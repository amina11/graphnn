import os
import shutil
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_curve, precision_recall_curve, accuracy_score, \
                            recall_score, precision_score, roc_auc_score, auc, average_precision_score
from matplotlib import pyplot as plt
from os.path import dirname, abspath, isfile
import itertools
from .model_attn_siamese import DeepAdr_SiameseTrf, DeepAdr_Transformer, FeatureEmbAttention
from .losses import ContrastiveLoss, CosEmbLoss
import torch
from torch import nn
import torch.multiprocessing as mp
import datetime


class ModelScore:
    def __init__(self, best_epoch_indx, s_auc, s_aupr, s_f1, s_precision, s_recall):
        self.best_epoch_indx = best_epoch_indx
        self.s_auc = s_auc
        self.s_aupr = s_aupr
        self.s_f1 = s_f1
        self.s_precision = s_precision
        self.s_recall = s_recall


    def __repr__(self):
        desc = " best_epoch_indx:{}\n auc:{} \n apur:{} \n f1:{} \n precision:{} \n recall:{} \n" \
               "".format(self.best_epoch_indx, self.s_auc, self.s_aupr, self.s_f1, self.s_precision, self.s_recall)
        return desc
    
class ModelScoreMultiClass:
    def __init__(self, best_epoch_indx, s_auc, s_f1_micro, s_f1_macro, s_precision_micro, s_precision_macro, s_recall_micro, s_recall_macro):
        self.best_epoch_indx = best_epoch_indx
        self.s_auc = s_auc
        self.s_f1_micro = s_f1_micro
        self.s_precision_micro = s_precision_micro
        self.s_recall_micro = s_recall_micro
        self.s_f1_macro = s_f1_macro
        self.s_precision_macro = s_precision_macro
        self.s_recall_macro = s_recall_macro


    def __repr__(self):
        desc = " best_epoch_indx:{}\n auc:{} \n f1_micro:{} \n f1_macro:{} \n precision_micro:{} \n precision_macro:{} \n recall_micro:{} \n recall_macro:{} \n" \
               "".format(self.best_epoch_indx, self.s_auc, self.s_f1_micro, self.s_f1_macro, self.s_precision_micro, self.s_precision_macro, self.s_recall_micro, self.s_recall_macro)
        return desc

def get_performance_results(similarity_type, target_dir, num_folds, dsettype):
    all_perf = {}
    num_metrics = 3 # number of metrics to focus on
    perf_dict = [{} for i in range(num_metrics)]  # track auc, aupr, f1 measure
    if dsettype == 'train':
        prefix = 'train_val'
    else:
        prefix = dsettype
    for fold_num in range(num_folds):

        fold_dir = os.path.join(target_dir,
                '{}'.format(prefix),
                'fold_{}'.format(fold_num))

        score_file = os.path.join(fold_dir, 'score_{}.pkl'.format(dsettype))

        if os.path.isfile(score_file):
            mscore = ReaderWriter.read_data(score_file)
            perf_dict[0]['fold{}'.format(fold_num)] = mscore.s_auc
            perf_dict[1]['fold{}'.format(fold_num)] = mscore.s_aupr
            perf_dict[2]['fold{}'.format(fold_num)] = mscore.s_f1
    perf_df = []
    for i in range(num_metrics):
        all_perf = perf_dict[i]
        all_perf_df = pd.DataFrame(all_perf, index=[similarity_type])
        median = all_perf_df.median(axis=1)
        mean = all_perf_df.mean(axis=1)
        stddev = all_perf_df.std(axis=1)
        all_perf_df['mean'] = mean
        all_perf_df['median'] = median
        all_perf_df['stddev'] = stddev
        perf_df.append(all_perf_df.sort_values('mean', ascending=False))
    return perf_df


def build_performance_dfs(similarity_types, target_dir, num_folds, dsettype):
    auc_df = pd.DataFrame()
    aupr_df = pd.DataFrame()
    f1_df = pd.DataFrame()
    target_dir = create_directory(target_dir, directory="parent")
    print(target_dir)
    for sim_type in similarity_types:
        s_auc, s_aupr, s_f1 = get_performance_results(sim_type, target_dir, num_folds, dsettype)
        auc_df = pd.concat([auc_df, s_auc], sort=True)
        aupr_df = pd.concat([aupr_df, s_aupr], sort=True)
        f1_df = pd.concat([f1_df, s_f1], sort=True)

    return auc_df, aupr_df, f1_df


class ReaderWriter(object):
    """class for dumping, reading and logging data"""
    def __init__(self):
        pass

    @staticmethod
    def read_or_dump_data(file_name, data_gen_fun, data_gen_params):
        if (isfile(file_name)):
            return ReaderWriter.read_data(file_name)
        else:
            data = data_gen_fun(*data_gen_params)
            ReaderWriter.dump_data(data, file_name)
            return data

    
    @staticmethod
    def dump_data(data, file_name, mode="wb"):
        """dump data by pickling
           Args:
               data: data to be pickled
               file_name: file path where data will be dumped
               mode: specify writing options i.e. binary or unicode
        """
        with open(file_name, mode) as f:
            pickle.dump(data, f)

    @staticmethod
    def read_data(file_name, mode="rb"):
        """read dumped/pickled data
           Args:
               file_name: file path where data will be dumped
               mode: specify writing options i.e. binary or unicode
        """
        with open(file_name, mode) as f:
            data = pickle.load(f)
        return(data)

    @staticmethod
    def dump_tensor(data, file_name):
        """
        Dump a tensor using PyTorch's custom serialization. Enables re-loading the tensor on a specific gpu later.
        Args:
            data: Tensor
            file_name: file path where data will be dumped
        Returns:
        """
        torch.save(data, file_name)

    @staticmethod
    def read_tensor(file_name, device):
        """read dumped/pickled data
           Args:
               file_name: file path where data will be dumped
               device: the gpu to load the tensor on to
        """
        data = torch.load(file_name, map_location=device)
        return data

    @staticmethod
    def write_log(line, outfile, mode="a"):
        """write data to a file
           Args:
               line: string representing data to be written out
               outfile: file path where data will be written/logged
               mode: specify writing options i.e. append, write
        """
        with open(outfile, mode) as f:
            f.write(line)

    @staticmethod
    def read_log(file_name, mode="r"):
        """write data to a file
           Args:
               line: string representing data to be written out
               outfile: file path where data will be written/logged
               mode: specify writing options i.e. append, write
        """
        with open(file_name, mode) as f:
            for line in f:
                yield line


# resolves relative paths, e.g. /path/to/../parent/dir/
def norm_join_paths(*paths):
    return os.path.normpath(os.path.join(*paths))
                
def create_directory(folder_name, directory="current"):
    """create directory/folder (if it does not exist) and returns the path of the directory
       Args:
           folder_name: string representing the name of the folder to be created
       Keyword Arguments:
           directory: string representing the directory where to create the folder
                      if `current` then the folder will be created in the current directory
    """
    if directory == "current":
        path_current_dir = os.path.dirname(__file__)  # __file__ refers to utilities.py
    elif directory == "parent":
        path_current_dir = dirname(dirname(abspath(__file__)))
    else:
        path_current_dir = directory
    #print("path_current_dir", path_current_dir)
        
    path_new_dir = os.path.normpath(os.path.join(path_current_dir, folder_name))
    if not os.path.exists(path_new_dir):
        os.makedirs(path_new_dir)
    return(path_new_dir)


def get_device(to_gpu, index=0):
    is_cuda = torch.cuda.is_available()
    if(is_cuda and to_gpu):
        target_device = 'cuda:{}'.format(index)
    else:
        target_device = 'cpu'
    return torch.device(target_device)


def report_available_cuda_devices():
    if(torch.cuda.is_available()):
        n_gpu = torch.cuda.device_count()
        print('number of GPUs available:', n_gpu)
        for i in range(n_gpu):
            print("cuda:{}, name:{}".format(i, torch.cuda.get_device_name(i)))
            device = torch.device('cuda', i)
            get_cuda_device_stats(device)
            print()
    else:
        print("no GPU devices available!!")

def get_cuda_device_stats(device):
    print('total memory available:', torch.cuda.get_device_properties(device).total_memory/(1024**3), 'GB')
    print('total memory allocated on device:', torch.cuda.memory_allocated(device)/(1024**3), 'GB')
    print('max memory allocated on device:', torch.cuda.max_memory_allocated(device)/(1024**3), 'GB')
    print('total memory cached on device:', torch.cuda.memory_reserved(device)/(1024**3), 'GB')
    print('max memory cached  on device:', torch.cuda.max_memory_reserved(device)/(1024**3), 'GB')

def get_interaction_stat(matrix):
    w, h = matrix.shape
    totalnum_elements = w*h
    nonzero_elem = np.count_nonzero(matrix)
    zero_elem = totalnum_elements - nonzero_elem
    print('number of rows: {}, cols: {}'.format(w, h))
    print('total number of elements', totalnum_elements)
    print('number of nonzero elements', nonzero_elem)
    print('number of zero elements', zero_elem)
    print('diagnoal elements ', np.diag(matrix))

def perfmetric_report(pred_target, ref_target, probscore, epoch, outlog, multi_class='raise'):
    lsep = "\n"
    report = "Epoch: {}".format(epoch) + lsep
    report += "Classification report on all events:" + lsep
    report += str(classification_report(ref_target, pred_target)) + lsep
    report += "macro f1:" + lsep
    macro_f1 = f1_score(ref_target, pred_target, average='macro')
    report += str(macro_f1) + lsep
    report += "micro f1:" + lsep
    micro_f1 = f1_score(ref_target, pred_target, average='micro')
    report += str(micro_f1) + lsep
    report += "accuracy:" + lsep
    accuracy = accuracy_score(ref_target, pred_target)
    report += str(accuracy) + lsep
        
    if (multi_class != "raise"):
        s_auc = roc_auc_score(ref_target, probscore, multi_class=multi_class)
        
        s_recall_micro = recall_score(ref_target, pred_target, average='micro')
        s_recall_macro = recall_score(ref_target, pred_target, average='macro')
        
        s_precision_micro = precision_score(ref_target, pred_target, average='micro')
        s_precision_macro = precision_score(ref_target, pred_target, average='macro')
        
        modelscore = ModelScoreMultiClass(epoch, s_auc, micro_f1, macro_f1, s_precision_micro, s_precision_macro, s_recall_micro, s_recall_macro)
        
    else:
        s_auc = roc_auc_score(ref_target, probscore)
#         report += "AUC:\n" + str(s_auc) + lsep
        precision_scores, recall_scores, __ = precision_recall_curve(ref_target, probscore)
        s_aupr = auc(recall_scores, precision_scores)
        report += "AUPR:\n" + str(s_aupr) + lsep
        s_f1 = f1_score(ref_target, pred_target)
        report += "binary f1:\n" + str(s_f1) + lsep
        s_recall = recall_score(ref_target, pred_target)
        s_precision = precision_score(ref_target, pred_target)
        
        modelscore = ModelScore(epoch, s_auc, s_aupr, s_f1, s_precision, s_recall)
    
    report += "AUC:\n" + str(s_auc) + lsep
    report += "-"*30 + lsep

    ReaderWriter.write_log(report, outlog)
    return modelscore


def plot_precision_recall_curve(ref_target, prob_poslabel, figname, outdir):
    pr, rec, thresholds = precision_recall_curve(ref_target, prob_poslabel)
    avg_precision = average_precision_score(ref_target, prob_poslabel)
    thresholds[0] = 1
    plt.figure(figsize=(9, 6))
    plt.plot(rec, pr, 'b+', label=f'Average Precision (AP):{avg_precision:.2}')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs. recall curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(outdir, os.path.join('precisionrecall_curve_{}'.format(figname) + ".pdf")))
    plt.close()


def plot_roc_curve(ref_target, prob_poslabel, figname, outdir):
    fpr, tpr, thresholds = roc_curve(ref_target, prob_poslabel)
    thresholds[0] = 1
    plt.figure(figsize=(9, 6))
    plt.plot(fpr, tpr, 'b+', label='TPR vs FPR')
    plt.plot(fpr, thresholds, 'r-', label='thresholds')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(outdir, os.path.join('roc_curve_{}'.format(figname) + ".pdf")))
    plt.close()

def plot_loss(epoch_loss_avgbatch, wrk_dir):
    dsettypes = epoch_loss_avgbatch.keys()
    for dsettype in dsettypes:
        plt.figure(figsize=(9, 6))
        plt.plot(epoch_loss_avgbatch[dsettype], 'r')
        plt.xlabel("number of epochs")
        plt.ylabel("negative loglikelihood cost")
        plt.legend(['epoch batch average loss'])
        plt.savefig(os.path.join(wrk_dir, os.path.join(dsettype + ".pdf")))
        plt.close()


def plot_xy(x, y, xlabel, ylabel, legend, fname, wrk_dir):
    plt.figure(figsize=(9, 6))
    plt.plot(x, y, 'r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend([legend])
    plt.savefig(os.path.join(wrk_dir, os.path.join(fname + ".pdf")))
    plt.close()

def find_youdenj_threshold(ref_target, prob_poslabel, fig_dir=None):
    fpr, tpr, thresholds = roc_curve(ref_target, prob_poslabel)
    s_auc = roc_auc_score(ref_target, prob_poslabel)
    thresholds[0] = 1
    plt.figure(figsize=(9, 6))
    plt.plot(fpr, tpr, 'b+', label=f'TPR vs FPR => AUC:{s_auc:.2}')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    youden_indx = np.argmax(tpr - fpr) # the index where the difference between tpr and fpr is max
    optimal_threshold = thresholds[youden_indx]
    plt.plot(fpr[youden_indx], tpr[youden_indx], marker='o', markersize=3, color="red", label=f'optimal probability threshold:{optimal_threshold:.2}')
    plt.legend(loc='best')
    if fig_dir:
        plt.savefig(f'{fig_dir}.pdf')
        plt.close()
    return fpr, tpr, thresholds, optimal_threshold

def analyze_precision_recall_curve(ref_target, prob_poslabel, fig_dir=None):
    pr, rec, thresholds = precision_recall_curve(ref_target, prob_poslabel)
    avg_precision = average_precision_score(ref_target, prob_poslabel)
    thresholds[0] = 1
    plt.figure(figsize=(9, 6))
    plt.plot(rec, pr, 'b+', label=f'Precision vs Recall => Average Precision (AP):{avg_precision:.2}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs. recall curve')
    indx = np.argmax(pr + rec)
    print('indx', indx)
    optimal_threshold = thresholds[indx]
    plt.plot(rec[indx], pr[indx], marker='o', markersize=3, color="red", label=f'optimal probability threshold:{optimal_threshold:.2}')
    plt.legend(loc='best')
    if fig_dir:
        plt.savefig(f'{fig_dir}.pdf')
        plt.close()
    return pr, rec, thresholds, optimal_threshold

def delete_directory(directory):
    if(os.path.isdir(directory)):
        shutil.rmtree(directory)


# code from keras https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py
def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return round(size,2), power_labels[n]+'bytes'

def add_weight_decay_except_attn(model_lst, l2_reg):
    decay, no_decay = [], []
    for m in model_lst:
        for name, param in m.named_parameters():
            if 'queryv' in name:
                no_decay.append(param)
            else: 
                decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_reg}]

def dump_dict_content(dsettype_content_map, dsettypes, desc, wrk_dir):
    for dsettype in dsettypes:
        path = os.path.join(wrk_dir, '{}_{}.pkl'.format(desc, dsettype))
        ReaderWriter.dump_data(dsettype_content_map[dsettype], path)
        
def get_random_fold(num_folds, random_seed=42):
    np.random.seed(random_seed)
    fold_num = np.random.randint(num_folds)
    return fold_num

def build_predictions_df(ids, true_class, pred_class, prob_scores):

    prob_scores_dict = {}
    for i in range (prob_scores.shape[-1]):
        prob_scores_dict[f'prob_score_class{i}'] = prob_scores[:, i]

    df_dict = {
        'id': ids,
        'true_class': true_class,
        'pred_class': pred_class
    }
    df_dict.update(prob_scores_dict)
    predictions_df = pd.DataFrame(df_dict)
    predictions_df.set_index('id', inplace=True)
    return predictions_df

def compute_numtrials(prob_interval_truemax, prob_estim):
    """ computes number of trials needed for random hyperparameter search
        see `algorithms for hyperparameter optimization paper
        <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__
        Args:
            prob_interval_truemax: float, probability interval of the true optimal hyperparam,
                i.e. within 5% expressed as .05
            prob_estim: float, probability/confidence level, i.e. 95% expressed as .95
    """
    n = np.log(1-prob_estim)/np.log(1-prob_interval_truemax)
    return(int(np.ceil(n))+1)

def get_saved_config(config_dir):
    options = ReaderWriter.read_data(os.path.join(config_dir, 'exp_options.pkl'))
    mconfig = ReaderWriter.read_data(os.path.join(config_dir, 'mconfig.pkl'))
    return mconfig, options


def get_index_argmax(score_matrix, target_indx):
    argmax_indx = np.argmax(score_matrix, axis=0)[target_indx]
    return argmax_indx