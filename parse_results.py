# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:58:29 2017

@author: DeepLearning
"""

import sys
import os
sys.path.insert(0,'Z:/Cristina/Section3/NME_DEC')
sys.path.insert(0,'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels')

import mxnet as mx
import numpy as np
import pandas as pd

from utilities import *
import data
import model
from autoencoder import AutoEncoderModel
from solver import Solver, Monitor
import logging

from sklearn.manifold import TSNE
from utilities import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import sklearn.neighbors 
import matplotlib.patches as mpatches
from sklearn.utils.linear_assignment_ import linear_assignment

try:
   import cPickle as pickle
except:
   import pickle
import gzip

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

#####################################################
#####################################################
## 1) read in the datasets both all NME (to do pretraining)
NME_nxgraphs = r'Z:\Cristina\Section3\breast_MR_NME_biological\NMEs_SER_nxgmetrics'
    
allNMEs_dynamic = pd.read_csv(os.path.join(NME_nxgraphs,'dyn_roi_records_allNMEs_descStats.csv'), index_col=0)
   
allNMEs_morphology = pd.read_csv(os.path.join(NME_nxgraphs,'morpho_roi_records_allNMEs_descStats.csv'), index_col=0)

allNMEs_texture = pd.read_csv(os.path.join(NME_nxgraphs,'text_roi_records_allNMEs_descStats.csv'), index_col=0)

allNMEs_stage1 = pd.read_csv(os.path.join(NME_nxgraphs,'stage1_roi_records_allNMEs_descStats.csv'), index_col=0)

# to load SERw matrices for all lesions
with gzip.open(os.path.join(NME_nxgraphs,'nxGdatafeatures_allNMEs_descStats.pklz'), 'rb') as fin:
    nxGdatafeatures = pickle.load(fin)

# to load discrall_dict dict for all lesions
with gzip.open(os.path.join(NME_nxgraphs,'nxGnormfeatures_allNMEs_descStats.pklz'), 'rb') as fin:
    discrall_dict_allNMEs = pickle.load(fin)         

allNME_featurenames = pd.read_csv(os.path.join(NME_nxgraphs,'named_nxGnormfeatures_allNMEs_descStats.csv'), index_col=0)

#########
# shape input (798L, 427L)    
nxGdiscfeatures = discrall_dict_allNMEs   
print('Loading {} all nxGdiscfeatures of size = {}'.format(nxGdiscfeatures.shape[0], nxGdiscfeatures.shape[1]) )

# shape input (798L, 427L)    
combX_allNME = np.concatenate((nxGdiscfeatures, allNMEs_dynamic.as_matrix(), allNMEs_morphology.as_matrix(), allNMEs_texture.as_matrix(), allNMEs_stage1.as_matrix()), axis=1)       
X_allNME_featurenames = np.concatenate((np.vstack(allNME_featurenames.columns),np.vstack(allNMEs_dynamic.columns),np.vstack(allNMEs_morphology.columns),np.vstack(allNMEs_texture.columns),np.vstack(allNMEs_stage1.columns)), axis=0).flatten()  

YnxG_allNME = np.asarray([nxGdatafeatures['roi_id'].values,
        nxGdatafeatures['classNME'].values,
        nxGdatafeatures['nme_dist'].values,
        nxGdatafeatures['nme_int'].values])

print('Loading {} all NME of size = {}'.format(combX_allNME.shape[0], combX_allNME.shape[1]) )
print('Loading all NME lables [label,BIRADS,dist,enh] of size = {}'.format(YnxG_allNME[0].shape[0])   )

# define variables for DEC 
roi_labels = YnxG_allNME[1]  
roi_labels = ['K' if rl=='U' else rl for rl in roi_labels]

## use y_dec to  minimizing KL divergence for clustering with known classes
ysup = ["{}_{}_{}".format(a, b, c) if b!='nan' else "{}_{}".format(a, c) for a, b, c in zip(YnxG_allNME[1], YnxG_allNME[2], YnxG_allNME[3])]
ysup = ['K'+rl[1::] if rl[0]=='U' else rl for rl in ysup]
classes = [str(c) for c in np.unique(ysup)]
numclasses = [i for i in range(len(classes))]
y_dec = []
for k in range(len(ysup)):
    for j in range(len(classes)):
        if(str(ysup[k])==classes[j]): 
            y_dec.append(numclasses[j])
y_dec = np.asarray(y_dec)

### Read input variables
# shape input (792, 523)     
nxGdiscfeatures = discrall_dict_allNMEs   
print('Loading {} leasions with nxGdiscfeatures of size = {}'.format(nxGdiscfeatures.shape[0], nxGdiscfeatures.shape[1]) )

print('Normalizing dynamic {} leasions with features of size = {}'.format(allNMEs_dynamic.shape[0], allNMEs_dynamic.shape[1]))
normdynamic = (allNMEs_dynamic - allNMEs_dynamic.mean(axis=0)) / allNMEs_dynamic.std(axis=0)
normdynamic.mean(axis=0)
print(np.min(normdynamic, 0))
print(np.max(normdynamic, 0))

print('Normalizing morphology {} leasions with features of size = {}'.format(allNMEs_morphology.shape[0], allNMEs_morphology.shape[1]))
normorpho = (allNMEs_morphology - allNMEs_morphology.mean(axis=0)) / allNMEs_morphology.std(axis=0)
normorpho.mean(axis=0)
print(np.min(normorpho, 0))
print(np.max(normorpho, 0))

print('Normalizing texture {} leasions with features of size = {}'.format(allNMEs_texture.shape[0], allNMEs_texture.shape[1]))
normtext = (allNMEs_texture - allNMEs_texture.mean(axis=0)) / allNMEs_texture.std(axis=0)
normtext.mean(axis=0)
print(np.min(normtext, 0))
print(np.max(normtext, 0))

print('Normalizing stage1 {} leasions with features of size = {}'.format(allNMEs_stage1.shape[0], allNMEs_stage1.shape[1]))
normstage1 = (allNMEs_stage1 - allNMEs_stage1.mean(axis=0)) / allNMEs_stage1.std(axis=0)
normstage1.mean(axis=0)
print(np.min(normstage1, 0))
print(np.max(normstage1, 0))    

# shape input (798L, 427L)    
combX_allNME = np.concatenate((nxGdiscfeatures, normdynamic.as_matrix(), normorpho.as_matrix(), normtext.as_matrix(), normstage1.as_matrix()), axis=1)       
YnxG_allNME = np.asarray([nxGdatafeatures['roi_id'].values,
        nxGdatafeatures['classNME'].values,
        nxGdatafeatures['nme_dist'].values,
        nxGdatafeatures['nme_int'].values])

print('Loading {} all NME of size = {}'.format(combX_allNME.shape[0], combX_allNME.shape[1]) )
print('Loading all NME lables [label,BIRADS,dist,enh] of size = {}'.format(YnxG_allNME[0].shape[0]) )

'''
######################
## From Pre-train/fine tune the SAE
######################
'''
save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels'
input_size = combX_allNME.shape[1]
latent_size = [input_size/rxf for rxf in [25,15,10,5,2]]

# train/test splits (test is 10% of labeled data)
sep = int(combX_allNME.shape[0]*0.10)
X_val = combX_allNME[:sep]
y_val = YnxG_allNME[1][:sep]
X_train = combX_allNME[sep:]
y_train = YnxG_allNME[1][sep:]
batch_size = 125 # 160 32*5 = update_interval*5
X_val[np.isnan(X_val)] = 0.00001

allAutoencoders = []
for output_size in latent_size:
    # Train or Read autoencoder: interested in encoding/decoding the input nxg features into LD latent space        
    # optimized for clustering with DEC
    xpu = mx.cpu()
    ae_model = AutoEncoderModel(xpu, [X_train.shape[1],500,500,2000,output_size], pt_dropout=0.2)
    ##  After Pre-train and finetuuning on X_train
    ae_model.load( os.path.join(save_to,'SAE_zsize{}_wimgfeatures_descStats_zeromean.arg'.format(str(output_size))) ) 

    ##  Get train/valid error (for Generalization)
    print "Autoencoder Training error: %f"%ae_model.eval(X_train)
    print "Autoencoder Validation error: %f"%ae_model.eval(X_val)
    # put useful metrics in a dict
    outdict = {'Training set': ae_model.eval(X_train),
               'Testing set': ae_model.eval(X_val),
               'output_size': output_size,
               'sep': sep}

    allAutoencoders.append(outdict)


'''
######################
## Visualize the reconstructed inputs and the encoded representations.
######################
'''
# train/test loss value o
dfSAE_perf = pd.DataFrame()
for SAE_perf in allAutoencoders:
    dfSAE_perf = dfSAE_perf.append( pd.DataFrame({'Reconstruction Error': pd.Series(SAE_perf)[0:2], 'train/validation':pd.Series(SAE_perf)[0:2].index, 'compressed size': SAE_perf['output_size']}) ) 

sns.set_style("darkgrid")
f, ax = plt.subplots(figsize=(10, 4))
sns.set_color_codes("pastel")
axSAE_perf = sns.pointplot(x="compressed size", y="Reconstruction Error", hue="train/validation", data=dfSAE_perf,  
                           markers=["x","o"], linestyles=["--","-"])  
aa=ax.set_xticklabels(['20x',"15x","10x","5x","2x"],fontsize=12)
ax.set_xlabel('compresed size',fontsize=14)
ax.set_ylabel('mean Reconstruccion Loss',fontsize=14)
ax.legend(loc="upper right",fontsize=15)


from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

# define variables for DEC 
roi_labels = YnxG_allNME[1]  
roi_labels = ['K' if rl=='U' else rl for rl in roi_labels]

# unbiased validation with held-out set
combX_allNME[np.isnan(combX_allNME)] = 0.1 
sep = int(combX_allNME.shape[0]*0.10)
X_val = combX_allNME[:sep]
y_val = roi_labels[:sep]

X_train = combX_allNME[sep:]
y_train = roi_labels[sep:]

# Classification and ROC analysis
datalabels = np.asarray(y_train)
dataspace = X_train
X = dataspace[datalabels!='K',:]
y = np.asarray(datalabels[datalabels!='K']=='M').astype(int)

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=5, random_state=3)
RFmodel = RandomForestClassifier(n_jobs=2, n_estimators=500, random_state=0, verbose=0)

# Evaluate a score by cross-validation
sns.set_style("whitegrid")
sns.set_color_codes("pastel")
figROCs = plt.figure(figsize=(8,8))    
axaroc = figROCs.add_subplot(1,1,1)              
tprs = []; aucs = []
mean_fpr_OrigX = np.linspace(0, 1, 100)
cvi = 0
for train, test in cv.split(X, y):
    probas = RFmodel.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas[:, 1])
    # to create an ROC with 100 pts
    tprs.append(interp(mean_fpr_OrigX, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    # plot
    #axaroc = figROCs.add_subplot(1,1,1)
    #axaroc.plot(fpr, tpr, lw=1, alpha=0.5) # with label add: label='cv %d, AUC %0.2f' % (cvi, roc_auc)
    cvi += 1

axaroc.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b',alpha=.8)
mean_tpr_OrigX = np.mean(tprs, axis=0)
mean_tpr_OrigX[-1] = 1.0
mean_auc_OrigX = auc(mean_fpr_OrigX, mean_tpr_OrigX)
std_auc_OrigX = np.std(aucs)
axaroc.plot(mean_fpr_OrigX, mean_tpr_OrigX, color='b',
         label=r'cv Train (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_OrigX, std_auc_OrigX),
         lw=2, alpha=1)     
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr_OrigX + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr_OrigX - std_tpr, 0)
axaroc.fill_between(mean_fpr_OrigX, tprs_lower, tprs_upper, color='grey', alpha=.15,
                 label=r'$\pm$ 1 std. dev.') 

'''
# plot AUC on validation set
'''
y_val_bin = (np.asarray(y_val)=='M').astype(int)
probas_val = RFmodel.fit(X, y).predict_proba(X_val)

# Compute ROC curve and area the curve
fpr_val_OrigX, tpr_val_OrigX, thresholds_val = roc_curve(y_val_bin, probas_val[:, 1])
auc_val_OrigX = auc(fpr_val_OrigX, tpr_val_OrigX)
axaroc.plot(fpr_val_OrigX, tpr_val_OrigX, color='g',linestyle='--',
            label=r'Test (AUC = %0.2f)' % (auc_val_OrigX),
             lw=2, alpha=1)     

axaroc.grid(False)
axaroc.set_xlabel('False Positive Rate',fontsize=18)
axaroc.set_ylabel('True Positive Rate',fontsize=18)
axaroc.set_title('ROC Original X HD labels={}, all features={} - Supervised cv RF classifier'.format(X.shape[0],X.shape[1]), fontsize=18)
axaroc.legend(loc="lower right",fontsize=18)

'''
####################################################################################################################################
## grid search results of dual-optimization: Z-space DEC LD and num_clusters + MLP classifier
#################################################################################################################################### 
'''  
# set config variables
from decModel_wimgF_dualopt_descStats import *
labeltype = 'wimgF_dualopt_descStats_saveparams' 
save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_wimgF_dualopt_descStats_saveparams'

# to load a prevously DEC model  
input_size = combX_allNME.shape[1]
latent_size = [input_size/rxf for rxf in [2,5,10,15]]
varying_mu = [int(np.round(var_mu)) for var_mu in np.linspace(3,12,10)]

scoresM = np.zeros((len(latent_size),len(varying_mu),5))
scoresM_titles=[]

sns.set_color_codes("pastel")

######################
# DEC: define num_centers according to clustering variable
######################   
# to load a prevously DEC model  
for ik,znum in enumerate(latent_size):
    cvRForigXAUC = []
    initAUC = []
    valAUC = []
    cvRFZspaceAUC = [] 
    normalizedMI = []
    for ic,num_centers in enumerate(varying_mu): 
        X = combX_allNME
        y = roi_labels
        y_train_roi_labels = np.asarray(y)

        print('Loading autoencoder of znum = {}, mu = {} , post training DEC results'.format(znum,num_centers))
        dec_model = DECModel(mx.cpu(), X, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 

        with gzip.open(os.path.join(save_to,'dec_model_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'rb') as fu:
            dec_model = pickle.load(fu)
          
        with gzip.open(os.path.join(save_to,'outdict_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'rb') as fu:
            outdict = pickle.load(fu)
        
        print('DEC train init AUC = {}'.format(outdict['meanAuc_cv'][0]))
        max_meanAuc_cv = outdict['meanAuc_cv'][-1]
        indmax_meanAuc_cv = outdict['meanAuc_cv'].index(max_meanAuc_cv)
        print r'DEC train max meanAuc_cv = {} $\pm$ {}'.format(max_meanAuc_cv,dec_model['std_auc'][indmax_meanAuc_cv])
        print('DEC validation AUC at max meanAuc_cv = {}'.format(outdict['auc_val'][indmax_meanAuc_cv]))
        
        #####################
        # extract Z-space from optimal DEC model
        #####################
        # saved output results
        dec_args_keys = ['encoder_1_bias', 'encoder_3_weight', 'encoder_0_weight', 
        'encoder_0_bias', 'encoder_2_weight', 'encoder_1_weight', 
        'encoder_3_bias', 'encoder_2_bias']
        dec_args = {key: v for key, v in dec_model.items() if key in dec_args_keys}
        dec_args['dec_mubestacci'] = dec_model['dec_mu']
        
        N = X.shape[0]
        all_iter = mx.io.NDArrayIter({'data': X}, batch_size=X.shape[0], shuffle=False,
                                                  last_batch_handle='pad')   
        ## extract embedded point zi 
        mxdec_args = {key: mx.nd.array(v) for key, v in dec_args.items() if key != 'dec_mubestacci'}                           
        aDEC = DECModel(mx.cpu(), X, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 
        
        # gather best-Zspace or dec_model['zbestacci']
        zbestacci = model.extract_feature(aDEC.feature, mxdec_args, None, all_iter, X.shape[0], aDEC.xpu).values()[0]      
        zbestacci = dec_model['zbestacci']
        
        # compute model-based best-pbestacci or dec_model['pbestacci']
        pbestacci = np.zeros((zbestacci.shape[0], dec_model['num_centers']))
        aDEC.dec_op.forward([zbestacci, dec_args['dec_mubestacci'].asnumpy()], [pbestacci])
        pbestacci = dec_model['pbestacci']
        
        # pool Z-space variables
        datalabels = np.asarray(y)
        dataZspace = np.concatenate((zbestacci, pbestacci), axis=1) 

        #####################
        # unbiased assessment: SPlit train/held-out test
        #####################
        # to compare performance need to discard unkown labels, only use known labels (only B or M)
        Z = dataZspace[datalabels!='K',:]
        y = datalabels[datalabels!='K']
      
        print '\n... MLP fully coneected layer trained on Z_train tested on Z_test' 
        sep = int(X.shape[0]*0.10)
        Z_test = Z[:sep]
        yZ_test = np.asanyarray(y[:sep]=='M').astype(int) 
        Z_train = Z[sep:]
        yZ_train = np.asanyarray(y[sep:]=='M').astype(int) 
       
        # We’ll load MLP using MXNet’s symbolic interface
        dataMLP = mx.sym.Variable('data')
        # MLP: two fully connected layers with 128 and 32 neurons each. 
        fc1  = mx.sym.FullyConnected(data=dataMLP, num_hidden = 128)
        act1 = mx.sym.Activation(data=fc1, act_type="relu")
        fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 32)
        act2 = mx.sym.Activation(data=fc2, act_type="relu")
        # data has 2 classes
        fc3  = mx.sym.FullyConnected(data=act2, num_hidden=2)
        # Softmax output layer
        mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
        # create a trainable module on CPU     
        batch_size = 50
        mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())
        # pass train/test data to allocate model (bind state)
        MLP_train_iter = mx.io.NDArrayIter(Z_train, yZ_train, batch_size, shuffle=False)
        mlp_model.bind(MLP_train_iter.provide_data, MLP_train_iter.provide_label)
        mlp_model.init_params()   
        mlp_model.init_optimizer()
        mlp_model_params = mlp_model.get_params()[0]
        
        # update parameters based on optimal found during cv Training
        from mxnet import ndarray
        params_dict = ndarray.load(os.path.join(save_to,'mlp_model_params_z{}_mu{}.arg'.format(znum,num_centers)))
        arg_params = {}
        aux_params = {}
        for k, value in params_dict.items():
            arg_type, name = k.split(':', 1)
            if arg_type == 'arg':
                arg_params[name] = value
            elif arg_type == 'aux':
                aux_params[name] = value
            else:
                raise ValueError("Invalid param file ")

        # order of params: [(128L, 266L),(128L,),(32L, 128L),(32L,),(2L, 32L),(2L,)]
        # organize weights and biases
        l1=[v.asnumpy().shape for k,v in mlp_model_params.iteritems()]
        k1=[k for k,v in mlp_model_params.iteritems()]
        l2=[v.asnumpy().shape for k,v in arg_params.iteritems()]
        k2=[k for k,v in arg_params.iteritems()]

        for ikparam,sizeparam in enumerate(l1):
            for jkparam,savedparam in enumerate(l2):
                if(sizeparam == savedparam):
                    #print('updating layer parameters: {}'.format(savedparam))
                    mlp_model_params[k1[ikparam]] = arg_params[k2[jkparam]]
        # upddate model parameters
        mlp_model.set_params(mlp_model_params, aux_params)
        
        #####################
        # ROC: Z-space MLP fully coneected layer for classification
        #####################
        figROCs = plt.figure(figsize=(18,6))    
        # Run classifier with cross-validation and plot ROC curves
        cv = StratifiedKFold(n_splits=5,random_state=3)
        # Evaluate a score by cross-validation
        tprs_train = []; aucs_train = []
        tprs_val = []; aucs_val = []
        mean_fpr = np.linspace(0, 1, 100)
        cvi = 0
        for train, test in cv.split(Z_train, yZ_train):
            ############### on train
            MLP_train_iter = mx.io.NDArrayIter(Z_train[train], yZ_train[train], batch_size)  
            # prob[i][j] is the probability that the i-th validation contains the j-th output class.
            prob_train = mlp_model.predict(MLP_train_iter)
            # Compute ROC curve and area the curve
            fpr_train, tpr_train, thresholds_train = roc_curve(yZ_train[train], prob_train.asnumpy()[:,1])
            # to create an ROC with 100 pts
            tprs_train.append(interp(mean_fpr, fpr_train, tpr_train))
            tprs_train[-1][0] = 0.0
            roc_auc = auc(fpr_train, tpr_train)
            aucs_train.append(roc_auc)
            
            ############### on validation
            MLP_val_iter = mx.io.NDArrayIter(Z_train[test], yZ_train[test], batch_size)    
            # prob[i][j] is the probability that the i-th validation contains the j-th output class.
            prob_val = mlp_model.predict(MLP_val_iter)
            # Compute ROC curve and area the curve
            fpr_val, tpr_val, thresholds_val = roc_curve(yZ_train[test], prob_val.asnumpy()[:,1])
            # to create an ROC with 100 pts
            tprs_val.append(interp(mean_fpr, fpr_val, tpr_val))
            tprs_val[-1][0] = 0.0
            roc_auc = auc(fpr_val, tpr_val)
            aucs_val.append(roc_auc)
            # plot
            #axaroc.plot(fpr, tpr, lw=1, alpha=0.6) # with label add: label='cv %d, AUC %0.2f' % (cvi, roc_auc)
            cvi += 1
           
        # plot for cv Train
        axaroc_train = figROCs.add_subplot(1,3,1)
        # add 50% or chance line
        axaroc_train.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b', alpha=.9)
        # plot mean and +- 1 -std as fill area
        mean_tpr_train = np.mean(tprs_train, axis=0)
        mean_tpr_train[-1] = 1.0
        mean_auc_train = auc(mean_fpr, mean_tpr_train)
        std_auc_train = np.std(aucs_train)
        axaroc_train.plot(mean_fpr, mean_tpr_train, color='b',
                    label=r'cv Train (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_train, std_auc_train),lw=3, alpha=1)     
        std_tpr = np.std(tprs_train, axis=0)
        tprs_upper = np.minimum(mean_tpr_train + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr_train - std_tpr, 0)
        axaroc_train.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.') 
        # set labels
        axaroc_train.set_xlabel('False Positive Rate',fontsize=16)
        axaroc_train.set_ylabel('True Positive Rate',fontsize=16)
        axaroc_train.set_title('Unsupervised DEC + cv MLP classifier Zspace dim={}'.format(Z.shape[1]),fontsize=18)
        axaroc_train.legend(loc="lower right",fontsize=16)
        plt.show()
        
        # plot for cv val
        axaroc_val = figROCs.add_subplot(1,3,2)
        # add 50% or chance line
        axaroc_val.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b', alpha=.9)
        # plot mean and +- 1 -std as fill area
        mean_tpr_val = np.mean(tprs_val, axis=0)
        mean_tpr_val[-1] = 1.0
        mean_auc_val = auc(mean_fpr, mean_tpr_val)
        std_auc_val = np.std(aucs_val)
        axaroc_val.plot(mean_fpr, mean_tpr_val, color='g',
                    label=r'cv Val (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_val, std_auc_val),lw=3, alpha=1)     
        std_tpr = np.std(tprs_val, axis=0)
        tprs_upper = np.minimum(mean_tpr_val + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr_val - std_tpr, 0)
        axaroc_val.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.') 
        # set labels
        axaroc_val.set_xlabel('False Positive Rate',fontsize=16)
        axaroc_val.set_ylabel('True Positive Rate',fontsize=16)
        #axaroc_val.set_title('Unsupervised DEC + cv MLP classifier Zspace dim={}'.format(Z.shape[1]),fontsize=14)
        axaroc_val.legend(loc="lower right",fontsize=16)
        plt.show()
        
        ################
        # plot AUC on heldout set
        ################
        MLP_heldout_iter = mx.io.NDArrayIter(Z_test, None, batch_size)   
        probas_heldout = mlp_model.predict(MLP_heldout_iter)
           
         # plot for cv val
        axaroc_test = figROCs.add_subplot(1,3,3)
        # add 50% or chance line
        axaroc_test.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b', alpha=.9)
        # Compute ROC curve and area the curve
        fpr_test, tpr_test, thresholds_test = roc_curve(yZ_test, probas_heldout.asnumpy()[:, 1])
        auc_test = auc(fpr_test, tpr_test)
        axaroc_test.plot(fpr_test, tpr_test, color='r',
                    label=r'Test (AUC = %0.2f)' % (auc_test),lw=3, alpha=1)     
        # set labels            
        axaroc_test.set_xlabel('False Positive Rate',fontsize=16)
        axaroc_test.set_ylabel('True Positive Rate',fontsize=16)
        #axaroc.set_title('ROC LD DEC optimized space={}, all features={} - Unsupervised DEC + cv MLP classifier'.format(Z.shape[0],Z.shape[1]),fontsize=18)
        axaroc_test.legend(loc="lower right",fontsize=16)
        plt.show()
    
        ############# append to 
        cvRForigXAUC.append(0.67)
        initAUC.append(dec_model['meanAuc_cv'][0])
        cvRFZspaceAUC.append(mean_auc_val)
        valAUC.append(auc_test)
        
        scoresM[ik,ic,0] = mean_auc_train
        scoresM_titles.append("DEC cv mean_auc_train")
        scoresM[ik,ic,1] = std_auc_train
        scoresM_titles.append("DEC cv std_auc_train")    
        scoresM[ik,ic,2] = mean_auc_val
        scoresM_titles.append("DEC cv mean_auc_val")
        scoresM[ik,ic,3] = std_auc_val
        scoresM_titles.append("DEC cv std_auc_val")       
        scoresM[ik,ic,4] = auc_test
        scoresM_titles.append("DEC heal-out test AUC")        
 
        
    # plot latent space Accuracies vs. original
    colors = plt.cm.jet(np.linspace(0, 1, 16))
    fig2 = plt.figure(figsize=(12,6))
    #ax2 = plt.axes()
    sns.set_context("notebook")
    ax1 = fig2.add_subplot(2,1,1)
    ax1.plot(varying_mu, cvRFZspaceAUC, color=colors[0], ls=':', label='cvRFZspaceAUC')
    ax1.plot(varying_mu, valAUC, color=colors[2], label='valAUC')
    ax1.plot(varying_mu, initAUC, color=colors[4], label='initAUC')
    ax1.plot(varying_mu, cvRForigXAUC, color=colors[6], label='cvRForigXAUC')
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':16})
    
    
'''
####################################################################################################################################
## Plot grid search results
#################################################################################################################################### 
'''  
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FixedLocator, FormatStrFormatter
import matplotlib.cm

figscoresM, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18)) 
for k,ax in enumerate(axes.flat):
    im = ax.imshow(scoresM[:,:,k], cmap='BuPu_r', interpolation='nearest')
    ax.grid(False)
    for u in range(len(latent_size)):        
        for v in range(len(varying_mu)):
            ax.text(v,u,'{:.2f}'.format(scoresM[u,v,k]), color=np.array([0.05,0.15,0.15,1]),
                         fontdict={'weight': 'bold', 'size': 14})
    # set ticks
    ax.xaxis.set_major_locator(FixedLocator(np.linspace(0,9,10)))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    
    mu_labels = [str(mu) for mu in varying_mu]
    ax.set_xticklabels(mu_labels, minor=False,fontsize=16)
    ax.yaxis.set_major_locator(FixedLocator(np.linspace(0,3,4)))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    
    znum_labels = ['2x','5x','10x','15x',] #[str(znum) for znum in latent_size]
    ax.set_yticklabels(znum_labels, minor=False,fontsize=16)
    ax.xaxis.set_label('latent space reduction')
    ax.yaxis.set_label('# cluster centroids')
    ax.set_title(scoresM_titles[k],fontsize=16)


## add colorbar
mins=[]; maxs=[]
# find min and max for scaling
for k in range(3):
    mins.append( np.min(scoresM[:,:,k]) )
    maxs.append( np.max(scoresM[:,:,k]) )
    
minsi = np.min(mins)
maxsi = np.max(maxs)
figscoresM.subplots_adjust(right=0.8)
cbar_ax = figscoresM.add_axes([0.85, 0.15, 0.05, 0.7])
im = ax.imshow(scoresM[:,:,0], cmap='BuPu_r', interpolation='nearest')
figscoresM.colorbar(im, cax=cbar_ax)
#figscoresM.savefig('Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels\\scoresM_AUC_{}_final.pdf'.format(labeltype), bbox_inches='tight')    

## plot line plots
input_size = combX_allNME.shape[1]
print "original input space = %d" % input_size 
dict_aucZlatent = pd.DataFrame() 
for k,znum in enumerate(latent_size):
    for l,num_c in enumerate(varying_mu):
        dict_aucZlatent = dict_aucZlatent.append( pd.Series({'Zspacedim':znum, 
                                                             'Zspace_AUC_ROC': scoresM[k,l,0], 
                                                              'Zspace_test_AUC_ROC': scoresM[k,l,2], 
                                                             'num_clusters':num_c}), ignore_index=True)
fig2 = plt.figure(figsize=(20,10))
ax2 = plt.axes()
sns.set_context("notebook")  
sns.pointplot(x="num_clusters", y="Zspace_AUC_ROC", hue="Zspacedim", data=dict_aucZlatent, ax=ax2, size=0.05) 
sns.pointplot(x="num_clusters", y="Zspace_test_AUC_ROC", hue="Zspacedim", data=dict_aucZlatent, ax=ax2, size=0.05, markers=["x","x","x","x"],linestyles=["--","--","--","--"]) 
ax2.xaxis.set_label('# clusters')
ax2.yaxis.set_label('Zspace AUC ROC')
ax2.set_title('Zspace AUC ROC vs. number of clusters',fontsize=20)
#fig2.savefig('Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels\\Zspace AUC ROC_{}.pdf'.format(labeltype), bbox_inches='tight')    
plt.close(fig2)

'''
##################################################################
## Find best performing parameters
################################################################## 
'''
dict_aucZlatent = pd.DataFrame() 
for k,znum in enumerate(latent_size):
    for l,num_c in enumerate(varying_mu):
        dict_aucZlatent = dict_aucZlatent.append( pd.Series({'Zspacedim':znum, 
                                                             'Zspace_AUC_ROC': scoresM[k,l,0], 
                                                              'Zspace_test_AUC_ROC': scoresM[k,l,2], 
                                                             'num_clusters':num_c}), ignore_index=True)
fig2 = plt.figure(figsize=(20,10))
ax2 = plt.axes()
sns.set_context("notebook")  
sns.pointplot(x="num_clusters", y="Zspace_AUC_ROC", hue="Zspacedim", data=dict_aucZlatent, ax=ax2, size=0.05) 
sns.pointplot(x="num_clusters", y="Zspace_test_AUC_ROC", hue="Zspacedim", data=dict_aucZlatent, ax=ax2, size=0.05, markers=["x","x","x","x"],linestyles=["--","--","--","--"]) 
ax2.xaxis.set_label('# clusters')
ax2.yaxis.set_label('Zspace AUC ROC')
ax2.set_title('Zspace AUC ROC vs. number of clusters',fontsize=20)


# plot box plots
dict_aucZlatent['Zspacedim_cats'] = pd.Series(dict_aucZlatent['Zspacedim'], dtype="category")
dict_aucZlatent['num_clusters_cats'] = pd.Series(dict_aucZlatent['num_clusters'], dtype="category")
print dict_aucZlatent

sns.set_color_codes("pastel")
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(2,2,1)
sns.boxplot(y="Zspace_AUC_ROC", x="Zspacedim_cats", data=dict_aucZlatent, ax=ax1)
znum_labels = ['15x','10x','5x','2x'] 
ax1.set_xticklabels(znum_labels, minor=False,fontsize=12)
ax1.set_xlabel('Latent space reduction')
ax1.set_ylabel('cv Train AUC ROC across # of centroids')

ax2 = fig.add_subplot(2,2,3)
sns.boxplot(y="Zspace_AUC_ROC", x="num_clusters_cats", data=dict_aucZlatent, ax=ax2)
ax2.set_xlabel('# cluster centroids')
ax2.set_ylabel('cv Train AUC ROC across reduction ratios')

ax3 = fig.add_subplot(2,2,2)
sns.boxplot(y="Zspace_test_AUC_ROC", x="Zspacedim_cats", data=dict_aucZlatent, ax=ax3)
ax3.set_xticklabels(znum_labels, minor=False,fontsize=12)
ax3.set_xlabel('Latent space reduction')
ax3.set_ylabel('Test AUC ROC across # of centroids')

ax4 = fig.add_subplot(2,2,4)
sns.boxplot(y="Zspace_test_AUC_ROC", x="num_clusters_cats", data=dict_aucZlatent, ax=ax4)
ax4.set_xlabel('# cluster centroids')
ax4.set_ylabel('Test AUC ROC across reduction ratios')


'''
##################################################################
## find best performing by the average of broth train and test performance
################################################################## 
'''

max_aucZlatent = np.max(dict_aucZlatent[["Zspace_AUC_ROC", "Zspace_test_AUC_ROC"]].mean(axis=1))
indmax_meanAuc_cv = dict_aucZlatent[["Zspace_AUC_ROC", "Zspace_test_AUC_ROC"]].mean(axis=1) == max_aucZlatent
print "\n================== Best average train/test perfomance parameters:" 
bestperf_params = dict_aucZlatent[indmax_meanAuc_cv]
print bestperf_params

'''
##################################################################
## TSNE
################################################################## 
'''
Zspacedim_best = int(bestperf_params.iloc[0]['Zspacedim'])
num_clusters_best = 3 #int(bestperf_params.iloc[0]['num_clusters'])

print('Loading autoencoder of Zspacedim_best = {}, mu = {} , post training DEC results'.format(Zspacedim_best,num_clusters_best))
dec_model = DECModel(mx.cpu(), X, num_clusters_best, 1.0, Zspacedim_best, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 

with gzip.open(os.path.join(save_to,'dec_model_z{}_mu{}_{}.arg'.format(Zspacedim_best,num_clusters_best,labeltype)), 'rb') as fu:
    dec_model = pickle.load(fu)
  
with gzip.open(os.path.join(save_to,'outdict_z{}_mu{}_{}.arg'.format(Zspacedim_best,num_clusters_best,labeltype)), 'rb') as fu:
    outdict = pickle.load(fu)
    
#####################
# extract Z-space from optimal DEC model
#####################
# saved output results
X = combX_allNME
y = roi_labels

dec_args_keys = ['encoder_1_bias', 'encoder_3_weight', 'encoder_0_weight', 
'encoder_0_bias', 'encoder_2_weight', 'encoder_1_weight', 
'encoder_3_bias', 'encoder_2_bias']
dec_args = {key: v for key, v in dec_model.items() if key in dec_args_keys}
dec_args['dec_mubestacci'] = dec_model['dec_mu']

N = X.shape[0]
all_iter = mx.io.NDArrayIter({'data': X}, batch_size=X.shape[0], shuffle=False,
                                          last_batch_handle='pad')   
## extract embedded point zi 
mxdec_args = {key: mx.nd.array(v) for key, v in dec_args.items() if key != 'dec_mubestacci'}                           
aDEC = DECModel(mx.cpu(), X, num_clusters_best, 1.0, Zspacedim_best, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 

# organize weights and biases
l1=[v.asnumpy().shape for k,v in aDEC.ae_model.args.iteritems()]
k1=[k for k,v in aDEC.ae_model.args.iteritems()]
l2=[v.asnumpy().shape for k,v in mxdec_args.iteritems()]
k2=[k for k,v in mxdec_args.iteritems()]

for ikparam,sizeparam in enumerate(l1):
    for jkparam,savedparam in enumerate(l2):
        if(sizeparam == savedparam):
            #print('updating layer parameters: {}'.format(savedparam))
            aDEC.ae_model.args[k1[ikparam]] = mxdec_args[k2[jkparam]]

zbestacci = model.extract_feature(aDEC.feature, mxdec_args, None, all_iter, X.shape[0], aDEC.xpu).values()[0]      

# compute model-based best-pbestacci or dec_model['pbestacci']
pbestacci = np.zeros((zbestacci.shape[0], dec_model['num_centers']))
aDEC.dec_op.forward([zbestacci, dec_args['dec_mubestacci'].asnumpy()], [pbestacci])
#pbestacci = dec_model['pbestacci']

# pool Z-space variables
datalabels = np.asarray(y)
dataZspace = np.concatenate((zbestacci, pbestacci), axis=1) 

#####################
# unbiased assessment: SPlit train/held-out test
#####################
# to compare performance need to discard unkown labels, only use known labels (only B or M)
Z = dataZspace[datalabels!='K',:]
y = datalabels[datalabels!='K']
  
print '\n... MLP fully coneected layer trained on Z_train tested on Z_test' 
sep = int(X.shape[0]*0.10)
Z_test = Z[:sep]
yZ_test = np.asanyarray(y[:sep]=='M').astype(int) 
Z_train = Z[sep:]
yZ_train = np.asanyarray(y[sep:]=='M').astype(int) 
   
# We’ll load MLP using MXNet’s symbolic interface
dataMLP = mx.sym.Variable('data')
# MLP: two fully connected layers with 128 and 32 neurons each. 
fc1  = mx.sym.FullyConnected(data=dataMLP, num_hidden = 128)
act1 = mx.sym.Activation(data=fc1, act_type="relu")
fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 32)
act2 = mx.sym.Activation(data=fc2, act_type="relu")
# data has 2 classes
fc3  = mx.sym.FullyConnected(data=act2, num_hidden=2)
# Softmax output layer
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
# create a trainable module on CPU     
batch_size = 50
mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())
# pass train/test data to allocate model (bind state)
MLP_train_iter = mx.io.NDArrayIter(Z_train, yZ_train, batch_size, shuffle=False)
mlp_model.bind(MLP_train_iter.provide_data, MLP_train_iter.provide_label)
mlp_model.init_params()   
mlp_model_params = mlp_model.get_params()[0]

# update parameters based on optimal found during cv Training
from mxnet import ndarray
params_dict = ndarray.load(os.path.join(save_to,'mlp_model_params_z{}_mu{}.arg'.format(Zspacedim_best,num_clusters_best)))
arg_params = {}
aux_params = {}
for k, value in params_dict.items():
    arg_type, name = k.split(':', 1)
    if arg_type == 'arg':
        arg_params[name] = value
    elif arg_type == 'aux':
        aux_params[name] = value
    else:
        raise ValueError("Invalid param file ")

# order of params: [(128L, 266L),(128L,),(32L, 128L),(32L,),(2L, 32L),(2L,)]
# organize weights and biases
l1=[v.asnumpy().shape for k,v in mlp_model_params.iteritems()]
k1=[k for k,v in mlp_model_params.iteritems()]
l2=[v.asnumpy().shape for k,v in arg_params.iteritems()]
k2=[k for k,v in arg_params.iteritems()]

for ikparam,sizeparam in enumerate(l1):
    for jkparam,savedparam in enumerate(l2):
        if(sizeparam == savedparam):
            #print('updating layer parameters: {}'.format(savedparam))
            mlp_model_params[k1[ikparam]] = arg_params[k2[jkparam]]
# upddate model parameters
mlp_model.set_params(mlp_model_params, aux_params)

#####################
# ROC: Z-space MLP fully coneected layer for classification
#####################
figROCs = plt.figure(figsize=(12,4))    
# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=5,random_state=3)
# Evaluate a score by cross-validation
tprs_train = []; aucs_train = []
tprs_val = []; aucs_val = []
mean_fpr = np.linspace(0, 1, 100)
## to append pooled predictions
pooled_pred_train = pd.DataFrame()
pooled_pred_val = pd.DataFrame()
cvi = 0
for train, test in cv.split(Z_train, yZ_train):
    ############### on train
    MLP_train_iter = mx.io.NDArrayIter(Z_train[train], yZ_train[train], batch_size)  
    # prob[i][j] is the probability that the i-th validation contains the j-th output class.
    prob_train = mlp_model.predict(MLP_train_iter)
    # Compute ROC curve and area the curve
    fpr_train, tpr_train, thresholds_train = roc_curve(yZ_train[train], prob_train.asnumpy()[:,1])
    # to create an ROC with 100 pts
    tprs_train.append(interp(mean_fpr, fpr_train, tpr_train))
    tprs_train[-1][0] = 0.0
    roc_auc = auc(fpr_train, tpr_train)
    aucs_train.append(roc_auc)
    
    ############### on validation
    MLP_val_iter = mx.io.NDArrayIter(Z_train[test], yZ_train[test], batch_size)    
    # prob[i][j] is the probability that the i-th validation contains the j-th output class.
    prob_val = mlp_model.predict(MLP_val_iter)
    # Compute ROC curve and area the curve
    fpr_val, tpr_val, thresholds_val = roc_curve(yZ_train[test], prob_val.asnumpy()[:,1])
    # to create an ROC with 100 pts
    tprs_val.append(interp(mean_fpr, fpr_val, tpr_val))
    tprs_val[-1][0] = 0.0
    roc_auc = auc(fpr_val, tpr_val)
    aucs_val.append(roc_auc)
    # plot
    #axaroc.plot(fpr, tpr, lw=1, alpha=0.6) # with label add: label='cv %d, AUC %0.2f' % (cvi, roc_auc)
    cvi += 1
    ## appends
    pooled_pred_train = pooled_pred_train.append( pd.DataFrame({"labels":yZ_train[train],
                          "probC":prob_train.asnumpy()[:,1],
                          "probNC":prob_train.asnumpy()[:,0]}), ignore_index=True)

    pooled_pred_val = pooled_pred_val.append( pd.DataFrame({"labels":yZ_train[test],
                          "probC":prob_val.asnumpy()[:,1],
                          "probNC":prob_val.asnumpy()[:,0]}), ignore_index=True)
                          
# plot for cv Train
axaroc_train = figROCs.add_subplot(1,3,1)
# add 50% or chance line
axaroc_train.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b', alpha=.9)
# plot mean and +- 1 -std as fill area
mean_tpr_train = np.mean(tprs_train, axis=0)
mean_tpr_train[-1] = 1.0
mean_auc_train = auc(mean_fpr, mean_tpr_train)
std_auc_train = np.std(aucs_train)
axaroc_train.plot(mean_fpr, mean_tpr_train, color='b',
            label=r'cv Train (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_train, std_auc_train),lw=3, alpha=1)     
std_tpr = np.std(tprs_train, axis=0)
tprs_upper = np.minimum(mean_tpr_train + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr_train - std_tpr, 0)
axaroc_train.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.') 
# set labels
axaroc_train.set_xlabel('False Positive Rate',fontsize=16)
axaroc_train.set_ylabel('True Positive Rate',fontsize=16)
axaroc_train.set_title('Unsupervised DEC + cv MLP classifier Zspace dim={}'.format(Z.shape[1]),fontsize=18)
axaroc_train.legend(loc="lower right",fontsize=16)

# plot for cv val
axaroc_val = figROCs.add_subplot(1,3,2)
# add 50% or chance line
axaroc_val.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b', alpha=.9)
# plot mean and +- 1 -std as fill area
mean_tpr_val = np.mean(tprs_val, axis=0)
mean_tpr_val[-1] = 1.0
mean_auc_val = auc(mean_fpr, mean_tpr_val)
std_auc_val = np.std(aucs_val)
axaroc_val.plot(mean_fpr, mean_tpr_val, color='g',
            label=r'cv Val (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_val, std_auc_val),lw=3, alpha=1)     
std_tpr = np.std(tprs_val, axis=0)
tprs_upper = np.minimum(mean_tpr_val + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr_val - std_tpr, 0)
axaroc_val.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.') 
# set labels
axaroc_val.set_xlabel('False Positive Rate',fontsize=16)
axaroc_val.set_ylabel('True Positive Rate',fontsize=16)
#axaroc_val.set_title('Unsupervised DEC + cv MLP classifier Zspace dim={}'.format(Z.shape[1]),fontsize=14)
axaroc_val.legend(loc="lower right",fontsize=16)

################
# plot AUC on heldout set
################
MLP_heldout_iter = mx.io.NDArrayIter(Z_test, None, batch_size)   
probas_heldout = mlp_model.predict(MLP_heldout_iter)
   
# plot for cv val
axaroc_test = figROCs.add_subplot(1,3,3)
# add 50% or chance line
axaroc_test.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b', alpha=.9)
# Compute ROC curve and area the curve
fpr_test, tpr_test, thresholds_test = roc_curve(yZ_test, probas_heldout.asnumpy()[:, 1])
auc_test = auc(fpr_test, tpr_test)
axaroc_test.plot(fpr_test, tpr_test, color='r',
            label=r'Test (AUC = %0.2f)' % (auc_test),lw=3, alpha=1)     
# set labels            
axaroc_test.set_xlabel('False Positive Rate',fontsize=16)
axaroc_test.set_ylabel('True Positive Rate',fontsize=16)
#axaroc.set_title('ROC LD DEC optimized space={}, all features={} - Unsupervised DEC + cv MLP classifier'.format(Z.shape[0],Z.shape[1]),fontsize=18)
axaroc_test.legend(loc="lower right",fontsize=16)
plt.show()

### add original for comparison
figROCs = plt.figure(figsize=(5,5))    
axaroc = figROCs.add_subplot(1,1,1)
axaroc.plot(mean_fpr_OrigX, mean_tpr_OrigX, color='b',
         label=r'HD cv Train (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_OrigX, std_auc_OrigX),lw=3, ls='--', alpha=.8)     
axaroc.plot(fpr_val_OrigX, tpr_val_OrigX, color='g',
            label=r'HD Test (AUC = %0.2f)' % (auc_val_OrigX),lw=3, ls='--',  alpha=.8)   

axaroc.set_xlabel('False Positive Rate',fontsize=18)
axaroc.set_ylabel('True Positive Rate',fontsize=18)
axaroc.set_title('ROC LD DEC optimized space={}, all features={} - Unsupervised DEC + cv MLP classifier'.format(Z.shape[0],Z.shape[1]),fontsize=18)
axaroc.legend(loc="lower right",fontsize=18)
plt.show()

'''
######################
## plot tSNE
######################
'''
figtsne = plt.figure(figsize=(12,16))
axtsne = figtsne.add_subplot(1,1,1)
            
tsne = TSNE(n_components=2, perplexity=50, learning_rate=125,
     init='pca', random_state=0, verbose=2, method='exact')

Z_tsne = tsne.fit_transform(dataZspace)   
plot_embedding_unsuper_NMEdist_intenh(Z_tsne, dec_model['named_y'], axtsne, title='final tsne: Zspace_AUC_ROC cv Train={}, Test={} \n'.format(bestperf_params['Zspace_AUC_ROC'], bestperf_params['Zspace_test_AUC_ROC']), legend=True)


'''
######################
## Generalization index
######################
'''
save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels'
DECsave_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_wimgF_dualopt_descStats_saveparams'
labeltype = 'wimgF_dualopt_descStats_saveparams'

input_size = combX_allNME.shape[1]
output_size = input_size/2

# train/test splits (test is 10% of labeled data)
sep = int(combX_allNME.shape[0]*0.10)
X_val = combX_allNME[:sep]
y_val = YnxG_allNME[1][:sep]
X_train = combX_allNME[sep:]
y_train = YnxG_allNME[1][sep:]
batch_size = 125 # 160 32*5 = update_interval*5
X_val[np.isnan(X_val)] = 0.1

allAutoencoders = []
for num_c in varying_mu:
    # Train or Read autoencoder: interested in encoding/decoding the input nxg features into LD latent space        
    # optimized for clustering with DEC
    xpu = mx.cpu()
    ae_model = AutoEncoderModel(mx.cpu(), [X_train.shape[1],500,500,2000,output_size], pt_dropout=0.2)
    ##  After Pre-train and finetuuning on X_train
    ae_model.load( os.path.join(save_to,'SAE_zsize{}_wimgfeatures_descStats_zeromean.arg'.format(str(output_size))) ) 

    with gzip.open(os.path.join(DECsave_to,'dec_model_z{}_mu{}_{}.arg'.format(output_size,num_c,labeltype)), 'rb') as fu:
        dec_model = pickle.load(fu)
        
    dec_args_keys = ['encoder_1_bias', 'encoder_3_weight', 'encoder_0_weight', 
    'encoder_0_bias', 'encoder_2_weight', 'encoder_1_weight', 
    'encoder_3_bias', 'encoder_2_bias']
    dec_args = {key: v for key, v in dec_model.items() if key in dec_args_keys}
    dec_args['dec_mubestacci'] = dec_model['dec_mu']
    # extract only dec args (encoders of SAE)
    mxdec_args = {key: mx.nd.array(v) for key, v in dec_args.items() if key != 'dec_mubestacci'}                           

    # organize weights and biases
    l1=[v.asnumpy().shape for k,v in ae_model.args.iteritems()]
    k1=[k for k,v in ae_model.args.iteritems()]
    l2=[v.asnumpy().shape for k,v in mxdec_args.iteritems()]
    k2=[k for k,v in mxdec_args.iteritems()]
    
    for ikparam,sizeparam in enumerate(l1):
        for jkparam,savedparam in enumerate(l2):
            if(sizeparam == savedparam):
                #print('updating layer parameters: {}'.format(savedparam))
                ae_model.args[k1[ikparam]] = mxdec_args[k2[jkparam]]
    
    ##  (2) generalizability (G) which is defined as the ratio between training and validation loss:
    print "Autoencoder Training error: %f"%ae_model.eval(X_train)
    print "Autoencoder Validation error: %f"%ae_model.eval(X_val)
    # put useful metrics in a dict
    outdict = {'Training set': ae_model.eval(X_train),
               'Testing set': ae_model.eval(X_val),
               'num_clusters': num_c,
               'sep': sep}

    allAutoencoders.append(outdict)

######################
## Visualize the reconstructed inputs and the encoded representations.
######################
dfSAE_perf = pd.DataFrame()
for SAE_perf in allAutoencoders:
    dfSAE_perf = dfSAE_perf.append( pd.DataFrame({'Reconstruction Error': pd.Series(SAE_perf)[0:2],
                                                  'generalizability': pd.Series(SAE_perf)[1]/pd.Series(SAE_perf)[0],
                                                  'train/validation':pd.Series(SAE_perf)[0:2].index, 
                                                  'num_clusters': SAE_perf['num_clusters']}) ) 
sns.set_style("darkgrid")
f, ax = plt.subplots(figsize=(10, 4))
sns.set_color_codes("pastel")
#axSAE_perf = sns.pointplot(x="num_clusters", y="Reconstruction Error", hue="train/validation", data=dfSAE_perf,  
#                           markers=["x","o"], linestyles=["--","-"])  
#axSAE_perf = sns.pointplot(x="num_clusters", y="generalizability", hue="train/validation", color='black', data=dfSAE_perf)                             

aucZlatent_output_size = dict_aucZlatent[dict_aucZlatent['Zspacedim_cats']==output_size]
axSAE_perf = sns.pointplot(x="num_clusters", y="Zspace_AUC_ROC", color='red', linestyles=["--"], data=aucZlatent_output_size)                             
sns.pointplot(x="num_clusters", y="Zspace_test_AUC_ROC", color='purple', linestyles=["-."], data=aucZlatent_output_size, ax=axSAE_perf)                             

ax.set_xlabel('# clusters',fontsize=14)
ax.set_ylabel('Zspace mean AUC ROC',fontsize=14)
ax.legend(loc="upper right",fontsize=15)