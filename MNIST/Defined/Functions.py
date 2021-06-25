#----------------------------Reproducible----------------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import random as rn
import os

seed=0
os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)
rn.seed(seed)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf =tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

#tf.set_random_seed(seed)
tf.compat.v1.set_random_seed(seed)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

#K.set_session(sess)
#----------------------------Reproducible----------------------------------------------------------------------------------------

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#--------------------------------------------------------------------------------------------------------------------------------
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Activation, Dropout, Layer
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras import optimizers,initializers,constraints,regularizers
from keras import backend as K
from keras.callbacks import LambdaCallback,ModelCheckpoint
from keras.utils import plot_model

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,roc_curve, auc,roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder,label_binarize,label_binarize

import h5py
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from functools import reduce
import seaborn as sns
from scipy import interp

#--------------------------------------------------------------------------------------------------------------------------------
def show_data_figures(p_data,w=28,h=28,columns = 16,p_pad=0,p_w_pad=0.15, p_h_pad=-4):
    
    # It shows the figures of digits, the input digits are "matrix version". This is the simple displaying in this codes
    rows = math.ceil(len(p_data)/columns)
    fig_show_w=columns
    fig_show_h=rows
    fig=plt.figure(figsize=(fig_show_w, fig_show_h))
    for i in range(0, len(p_data)):
        fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        plt.imshow(p_data[i,:].reshape((w, h)),plt.cm.gray)
    plt.tight_layout(p_pad,p_w_pad, p_h_pad)
    plt.show()
    
#--------------------------------------------------------------------------------------------------------------------------------
def show_data_mix_figures(p_data,w=28,h=28,columns = 16,p_pad=0,p_w_pad=0.15, p_h_pad=-4):
    
    # It shows the figures of digits, the input digits are "matrix version". This is the simple displaying in this codes
    rows = math.ceil(len(p_data)/columns)
    fig_show_w=columns
    fig_show_h=rows
    fig=plt.figure(figsize=(fig_show_w, fig_show_h))
    for i in range(0, len(p_data)):
        fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        plt.imshow(p_data[i,:].reshape((w, h)),plt.cm.gray, interpolation="bicubic")
    plt.tight_layout(p_pad,p_w_pad, p_h_pad)
    plt.show()
    
def largest_indices_value(p_ary, p_n):
    flat = p_ary.flatten()
    indices__ = np.argpartition(flat, -p_n)[-p_n:]
    indices_ = indices__[np.argsort(-flat[indices__])]
    indices=np.unravel_index(indices_, p_ary.shape)
    indices_collect=[]
    indices_row=indices[0]
    indices_col=indices[1]
    indices_collect=[]
    for i in np.arange(p_n):
        indices_collect.append((indices_row[i],indices_col[i]))
    return indices_collect,indices,p_ary[indices]

def show_one_figure_with_keyfeature(p_feature_weights,p_key_feature_catch,w=92,h=112,p_pad=0,p_w_pad=0.15, p_h_pad=-4):
    plt.imshow(p_feature_weights.reshape((w, h)))
    for key_feature_catch_i in np.arange(len(p_key_feature_catch)):
        plt.scatter(p_key_feature_catch[key_feature_catch_i][1],p_key_feature_catch[key_feature_catch_i][0],s=1,color='r')
    plt.axis('off')
    plt.tight_layout(p_pad,p_w_pad, p_h_pad)
    plt.show()
    
def show_data_figures_with_keyfeature(data,p_key_feature_catch,w=92,h=112,columns = 16,p_pad=0,p_w_pad=0.15, p_h_pad=-4):
    
    # It shows the figures of digits, the input digits are "matrix version". This is the simple displaying in this codes
    rows = math.ceil(len(data)/columns)
    fig_show_w=columns
    fig_show_h=rows
    fig=plt.figure(figsize=(fig_show_w, fig_show_h))
    for i in range(0, len(data)):
        fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        plt.imshow(data[i,:].reshape((w, h)),plt.cm.gray)
        for key_feature_catch_i in np.arange(len(p_key_feature_catch)):
            plt.scatter(p_key_feature_catch[key_feature_catch_i][1],p_key_feature_catch[key_feature_catch_i][0],s=1,color='r')
    plt.tight_layout(p_pad,p_w_pad, p_h_pad)
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------------
def top_k_keep(p_arr_,p_top_k_):
    top_k_idx=p_arr_.argsort()[::-1][0:p_top_k_]
    top_k_value=p_arr_[top_k_idx]
    return np.where(p_arr_<top_k_value[-1],0,p_arr_)

#--------------------------------------------------------------------------------------------------------------------------------
def show_feature_selection(p_file_name,p_test_data,p_sample_number=40,p_key_number=36):
    file = h5py.File(p_file_name,'r') 
    data = file['feature_selection']['feature_selection']['kernel:0']
    weight_top_k=top_k_keep(np.array(data),p_key_number)
    show_data_figures(np.dot(p_test_data[0:p_sample_number],np.diag(weight_top_k)))
    file.close()

#--------------------------------------------------------------------------------------------------------------------------------
'''
def top_k_keepWeights_1(p_arr_,p_top_k_):
    top_k_idx=p_arr_.argsort()[::-1][0:p_top_k_]
    top_k_value=p_arr_[top_k_idx]
    if np.sum(p_arr_>0)>p_top_k_:
        p_arr_=np.where(p_arr_<top_k_value[-1],0,1)
    else:
        p_arr_=np.where(p_arr_<=0,0,1) 
    return p_arr_
'''

def top_k_keepWeights_1(p_arr_,p_top_k_,p_ignore_equal=True):
    top_k_idx=p_arr_.argsort()[::-1][0:p_top_k_]
    top_k_value=p_arr_[top_k_idx]
    if np.sum(p_arr_>0)>p_top_k_:
        if p_ignore_equal:
            p_arr_=np.where(p_arr_<top_k_value[-1],0,1)
        else:
            p_arr_=np.where(p_arr_<=top_k_value[-1],0,1)
    else:
        p_arr_=np.where(p_arr_<=0,0,1) 
    return p_arr_

#--------------------------------------------------------------------------------------------------------------------------------
def hierarchy_top_k_keep(p_arr_,p_choose_top_k_,p_selection_hierarchy):

    if p_selection_hierarchy==1:
        top_k_idx=p_arr_.argsort()[::-1][0:p_choose_top_k_]
        top_k_value=p_arr_[top_k_idx]        
        return np.where(p_arr_<top_k_value[-1],0,p_arr_)
    elif p_selection_hierarchy>1:
        top_k_idx=p_arr_.argsort()[::-1][0:(p_selection_hierarchy-1)*p_choose_top_k_]
        top_k_value_1=p_arr_[top_k_idx]
        p_arr_=np.where(p_arr_<top_k_value_1[-1],p_arr_,0)
        
        top_k_idx=p_arr_.argsort()[::-1][0:p_choose_top_k_]
        top_k_value_2=p_arr_[top_k_idx]
        return np.where(p_arr_<top_k_value_2[-1],0,p_arr_)      

#--------------------------------------------------------------------------------------------------------------------------------
def show_hierarchy_feature_selection(p_file_name,p_test_data,p_selection_hierarchy=1,p_sample_number=40,p_key_number=36):
    file = h5py.File(p_file_name,'r') 
    data = file['feature_selection']['feature_selection']['kernel:0']
    weight_top_k=hierarchy_top_k_keep(np.array(data),p_key_number,p_selection_hierarchy)
    show_data_figures(np.dot(p_test_data[0:p_sample_number],np.diag(weight_top_k)))
    file.close()
    
#--------------------------------------------------------------------------------------------------------------------------------
def hierarchy_top_k_keepWeights_1(p_arr_,p_choose_top_k_,p_selection_hierarchy):

    if p_selection_hierarchy==1:
        top_k_idx=p_arr_.argsort()[::-1][0:p_choose_top_k_]
        top_k_value=p_arr_[top_k_idx]        
        if np.sum(p_arr_>0)>p_choose_top_k_:
            p_arr_=np.where(p_arr_<top_k_value[-1],0,1)
        else:
            p_arr_=np.where(p_arr_<=0,0,1) 
        return p_arr_   

    elif p_selection_hierarchy>1:
        top_k_idx=p_arr_.argsort()[::-1][0:(p_selection_hierarchy-1)*p_choose_top_k_]
        top_k_value_1=p_arr_[top_k_idx]  
        
        p_arr_=np.where(p_arr_<top_k_value_1[-1],p_arr_,0)

        top_k_idx=p_arr_.argsort()[::-1][0:p_choose_top_k_]
        top_k_value_2=p_arr_[top_k_idx]
        
        if np.sum(p_arr_>0)>p_choose_top_k_:
            p_arr_=np.where(p_arr_<top_k_value_2[-1],0,1)
        else:
            p_arr_=np.where(p_arr_<=0,0,1) 
        return p_arr_ 

#--------------------------------------------------------------------------------------------------------------------------------
def show_data_figures_with_hierarchy_keyfeature(p_data,p_key_feature_catch,w=28,h=28,columns = 20):
    
    # It shows the figures of digits, the input digits are "matrix version". This is the simple displaying in this codes
    rows = math.ceil(len(p_data)/columns)
    fig_show_w=columns
    fig_show_h=rows
    fig=plt.figure(figsize=(fig_show_w, fig_show_h))
    for i in range(0, len(p_data)):
        fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        plt.imshow(p_data[i,:].reshape((w, h)),plt.cm.gray)
        p_key_feature_catch_i=p_key_feature_catch[i]
        for key_feature_catch_i in np.arange(len(p_key_feature_catch_i)):
            plt.scatter(p_key_feature_catch_i[key_feature_catch_i][1],p_key_feature_catch_i[key_feature_catch_i][0],s=35,color='r')
    plt.tight_layout()
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------------
def ETree(p_train_feature,p_train_label,p_test_feature,p_test_label,p_seed):
    clf = ExtraTreesClassifier(n_estimators=50, random_state=p_seed)
    
    # Training
    clf.fit(p_train_feature, p_train_label)
    
    # Training accuracy
    print('Training accuracy：',clf.score(p_train_feature, np.array(p_train_label)))
    print('Training accuracy：',accuracy_score(np.array(p_train_label),clf.predict(p_train_feature)))
    #print('Training accuracy：',np.sum(clf.predict(p_train_feature)==np.array(p_train_label))/p_train_label.shape[0])

    # Testing accuracy
    print('Testing accuracy：',clf.score(p_test_feature, np.array(p_test_label)))
    print('Testing accuracy：',accuracy_score(np.array(p_test_label),clf.predict(p_test_feature)))
    #print('Testing accuracy：',np.sum(clf.predict(p_test_feature)==np.array(p_test_label))/p_test_label.shape[0])

#--------------------------------------------------------------------------------------------------------------------------------   
def compress_zero(p_data_matrix,p_key_feture_number):
    p_data_matrix_Results=[]
    for p_data_matrix_i in p_data_matrix:
        p_data_matrix_Results_i=[]
        for ele_i in p_data_matrix_i:
            if ele_i!=0:
                p_data_matrix_Results_i.append(ele_i)
        if len(p_data_matrix_Results_i)<p_key_feture_number:
            for add_i in np.arange(p_key_feture_number-len(p_data_matrix_Results_i)):
                p_data_matrix_Results_i.append(0)
        p_data_matrix_Results.append(p_data_matrix_Results_i)
    return np.array(p_data_matrix_Results)

#-------------------------------------------------------------------------------------------------------------------------------- 
def compress_zero_withkeystructure(p_data_matrix,p_selected_position,w=28,h=28):
    p_data_matrix_Results=[]
    for p_data_matrix_i in p_data_matrix:
        p_data_matrix_Results_i=[]
        p_data_matrix_i_=p_data_matrix_i.reshape(w,h)
        for selection_j in p_selected_position:
            p_data_matrix_Results_i.append(p_data_matrix_i_[selection_j])
        p_data_matrix_Results.append(p_data_matrix_Results_i)
    return np.array(p_data_matrix_Results)

#--------------------------------------------------------------------------------------------------------------------------------
def k_index_argsort(data, k, order='maxs'): 
    #return the k largest
    if (order=='maxs'):
        idx = np.argsort(data.ravel())[:-k-1:-1] 
    #return the k smallest
    elif(order=='mins'):
        idx = np.argsort(data.ravel())[0:k:] 
    else:
        print('Wrong input! Try again!')
    return np.column_stack(np.unravel_index(idx, data.shape)) 

#--------------------------------------------------------------------------------------------------------------------------------
def show_checking_key_features_on_testingsamples(data,p_key_feature_catch,w=112,h=92,columns = 20,p_pad=0,p_w_pad=0.15, p_h_pad=-4):
    
    # It shows the figures of digits, the input digits are "matrix version". This is the simple displaying in this codes
    rows = math.ceil(len(data)/columns)
    fig_show_w=columns
    fig_show_h=rows
    fig=plt.figure(figsize=(fig_show_w, fig_show_h))
    for i in range(0, len(data)):
        fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        plt.imshow(data[i,:].reshape((w, h)),plt.cm.gray)
        
        for key_feature_catch_i in p_key_feature_catch[i]:
            if (key_feature_catch_i[0]>0): 
                plt.scatter(key_feature_catch_i[1],key_feature_catch_i[0],s=35,color='r')
        #for key_feature_catch_i in np.arange(len(p_key_feature_catch[i])):
        #    if (p_key_feature_catch[i][key_feature_catch_i][1]>0):           
        #        plt.scatter(p_key_feature_catch[i][key_feature_catch_i][1],p_key_feature_catch[i][key_feature_catch_i][0],s=2,color='r')
    
    plt.tight_layout(p_pad,p_w_pad, p_h_pad)
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------------
def intersect_2Dlist(p_selected_genes_list,p_print=False):
    intersect_selected_genes=reduce(lambda x,y : set(x) & set(y), p_selected_genes_list)
    if p_print:
        print(intersect_selected_genes)
    return np.array(list(intersect_selected_genes))

#--------------------------------------------------------------------------------------------------------------------------------
def write_to_csv(p_data,p_path):
    dataframe = pd.DataFrame(p_data)
    dataframe.to_csv(p_path, mode='a',header=False,index=False,sep=',')

#--------------------------------------------------------------------------------------------------------------------------------
def draw_with_importance(p_importance,original_edge_width,original_edge_length,top_select):
    key_feature_catch,\
    key_feature_catch_orignal_form,\
    key_feature_value_catch=largest_indices_value(p_importance.reshape(original_edge_width,original_edge_length),\
                                                    top_select)

    p_feature_weights=p_importance
    p_key_feature_catch=key_feature_catch

    show_one_figure_with_keyfeature(p_feature_weights,p_key_feature_catch,w=original_edge_width,h=original_edge_length)
    
#--------------------------------------------------------------------------------------------------------------------------------   
def show_global_sub_curves(p_score_loss,p_choose_loss,p_epoches,p_models_used_show,\
                           p1_xlabel='Epoches',p1_ylabel='Losses',p1_title="Global-neural netwok",\
                           p2_xlabel='Epoches',p2_ylabel='Losses',p2_title="Sub-neural netwok"):

    plt.style.use('seaborn-dark')

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(24,8))

    ax1.grid(True,linestyle='--',linewidth = 2.5,zorder=-1,axis="y")

    ax1.set_xlabel(p1_xlabel, fontsize = 28)
    ax1.set_ylabel(p1_ylabel, fontsize = 28)
    ax1.tick_params(labelsize=28)
    ax1.set_title(p1_title,fontsize=28)

    ax2.grid(True,linestyle='--',linewidth = 2.5,zorder=-1,axis="y")
    ax2.set_xlabel(p2_xlabel, fontsize =28)
    ax2.set_ylabel(p2_ylabel, fontsize = 28)
    ax2.tick_params(labelsize=28)
    ax2.set_title(p2_title,fontsize=28)

    Markers=['o', '+', '2', 'v', 's','X','d','>','*']
    Colors=["orange","blue","darkcyan","fuchsia","chocolate","aqua","green","dodgerblue","red"]

    for i in np.arange(1,p_models_used_show+1):
        ax1.plot(p_epoches, p_score_loss[i-1], marker=Markers[i-1], c=Colors[i-1], mfc='w',ms=8)

    for i in np.arange(1,p_models_used_show+1):
        ax2.plot(p_epoches, p_choose_loss[i-1], marker=Markers[i-1], c=Colors[i-1], mfc='w',ms=8)

    plt.subplots_adjust(right=1,wspace =0.15,hspace =0)

    fig.legend(labels=["M "+str(i) for i in np.arange(1,p_models_used_show+1)],fontsize=28, loc='upper center', bbox_to_anchor=(0.49,1.02),ncol=10,handletextpad=0.1,columnspacing=0.3, fancybox=True,framealpha=0.1,shadow=True)

    plt.show()
    
#--------------------------------------------------------------------------------------------------------------------------------   
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


#--------------------------------------------------------------------------------------------------------------------------------
def draw_multi_class_roc(p_n_classes,p_y_test,p_y_pred_choose,p_title,p_lw=2):
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.style.use('seaborn-dark')

    fig = plt.figure(figsize = (10,6))

    plt.grid(True,linestyle='--',linewidth = 2.5,zorder=-1,axis="y")

    for i in range(p_n_classes):
        fpr[i], tpr[i], _ = roc_curve(p_y_test[:, i], p_y_pred_choose[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area（方法二） 
    fpr["micro"], tpr["micro"], _ = roc_curve(p_y_test.ravel(), p_y_pred_choose.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"]) 
    # Compute macro-average ROC curve and ROC area（方法一） 
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(p_n_classes)])) 
    # Then interpolate all ROC curves at this points 
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(p_n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i]) 
    
    # Finally average it and compute AUC 
    mean_tpr /= p_n_classes 
    fpr["macro"] = all_fpr 
    tpr["macro"] = mean_tpr 
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"]) 

    # Plot all ROC curves 
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4) 
    plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)

    colors =np.array(['blue', 'red', 'green', 'yellow'])
    for i, color in zip(range(p_n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,lw=p_lw,label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.title('Receiver operating characteristic for '+p_title,fontsize=18)

    plt.legend(prop = {'size':16},loc='upper center',ncol=2,fancybox=True,shadow=True,bbox_to_anchor=(0.5,1.35))
    plt.show()
    
    
