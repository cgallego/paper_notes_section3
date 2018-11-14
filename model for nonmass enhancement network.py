# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 14:07:42 2017

@author: DeepLearning
"""

import pandas as pd
import numpy as np
import os
import os.path
import shutil
import glob
import tempfile
import subprocess
import SimpleITK as sitk

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({'font.size': 8})

import numpy.ma as ma
from skimage.measure import find_contours, approximate_polygon
from scipy.spatial import Delaunay
from matplotlib.collections import LineCollection

# to save graphs
import six.moves.cPickle as pickle
import gzip
import networkx as nx
from skimage.morphology import skeletonize, skeletonize_3d
from scipy.linalg import det

import logging
import sys
sys.path.insert(0,'Z:\\Cristina\Section3\\breast_MR_NME_biological')

from query_localdatabase import *
from run_cad_pipeline_fdicoms import *

# Set the font dictionaries (for plot title and axis titles)
title_font = {'fontname':'Arial', 'size':'24', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
axis_font = {'fontname':'Arial', 'size':'16'}

data_loc = 'Z:\\Cristina\\Section3\\breast_MR_NME_pipeline'
processed_path = r'Z:\Cristina\Section3\breast_MR_NME_pipeline\processed_data' ##r'E:\Users\DeepLearning\outputs' ##
mha_data_loc= 'Z:\\Cristina\\mha'

path_rootFolder = 'Z:\\Cristina\\Section3\\breast_MR_NME_biological'# os.path.dirname(os.path.abspath(__file__)) #
processed_NMEs_path = r'Z:\Cristina\Section3\breast_MR_NME_biological\processed_NMEs'

lesion_id = 1
localdata = Querylocaldb()
dflesion  = localdata.querylocalDatabase_wRad(lesion_id)      
cond = dflesion[0]
lesion_record = dflesion[1]
roi_record = dflesion[2]
nmlesion_record = dflesion[3]
StudyID = lesion_record['cad_pt_no_txt']
AccessionN = lesion_record['exam_a_number_txt']
DynSeries_id = nmlesion_record['DynSeries_id']  
roi_id = roi_record['roi_id']
label = roi_record['roi_label']
c = roi_record['roi_centroid']
centroid = c[c.find("(")+1:c.find(")")].split(',')
zslice = int(roi_record['zslice'])
p1 = roi_record['patch_diag1']
patch_diag1 = p1[p1.find("(")+1:p1.find(")")].split(',')
patch_diag1 = [float(p) for p in patch_diag1]
p2 = roi_record['patch_diag2']
patch_diag2 = p2[p2.find("(")+1:p2.find(")")].split(',')
patch_diag2 = [float(p) for p in patch_diag2]    

print("====================")
print('StudyID: ', StudyID)
print('AccessionN: ', AccessionN)
print('DynSeries_id: ', DynSeries_id)
print('label: ', label)
print("====================")


#############################
###### 1) Accesing mc images and lesion prob maps
#############################
# get dynmic series info
precontrast_id = int(DynSeries_id) 
DynSeries_nums = [str(n) for n in range(precontrast_id,precontrast_id+5)]

print "Reading MRI volumes..."
DynSeries_imagefiles = []
mriVols = []
preCon_filename = '{}_{}_{}'.format(int(StudyID.zfill(4)),AccessionN,DynSeries_nums[0] )
glob_result = glob.glob(os.path.join(mha_data_loc,preCon_filename+'@*')) #'*':do not to know the exactly acquistion time
if glob_result != []:
    filename = glob_result[0]
# read Volumnes
DynSeries_imagefiles.append(filename)
mriVolDICOM = sitk.ReadImage(filename)
mriVols.append( sitk.GetArrayFromImage(sitk.Cast(mriVolDICOM,sitk.sitkFloat32)) )
mriVolSize = mriVolDICOM.GetSize()
print "MRI volumes Size = [%f,%f,%f]..." % mriVolSize
mriVolSpacing = mriVolDICOM.GetSpacing()
print "MRI volumes spacing = [%f,%f,%f]..." % mriVolSpacing
mriVolVoxratio = mriVolSpacing[2]/mriVolSpacing[0]        

for j in range(1,5):
    #the output mha:lesionid_patientid_access#_series#@acqusionTime.mha
    DynSeries_filename = '{}_{}_{}'.format(StudyID.zfill(4),AccessionN,DynSeries_nums[j] )

    #write log if mha file not exist             
    glob_result = glob.glob(os.path.join(processed_path,DynSeries_filename+'@*')) #'*':do not to know the exactly acquistion time
    if glob_result != []:
        filename = [name for name in glob_result if '_mc' in name][0] #glob_result[0]
        print filename

    # add side info from the side of the lesion
    DynSeries_imagefiles.append(filename)

    # read Volumnes
    mriVolDICOM = sitk.ReadImage(filename)
    mriVols.append( sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(DynSeries_imagefiles[j]),sitk.sitkFloat32)) )



print "Reading probability map... and define ROI"
probmap_filename = '{}_{}_lesion_segmentation_probability.mha'.format(StudyID.zfill(4),AccessionN)
probmap_filepath = os.path.join(processed_path,probmap_filename)
probmap = sitk.GetArrayFromImage(sitk.Cast( sitk.ReadImage(probmap_filepath), sitk.sitkFloat32)) 

mx_query = np.zeros(probmap.shape)
ext_x = [int(ex) for ex in [np.min([patch_diag1[0],patch_diag2[0]])-10,np.max([patch_diag1[0],patch_diag2[0]])+10] ] 
ext_y = [int(ey) for ey in [np.min([patch_diag1[1],patch_diag2[1]])-10,np.max([patch_diag1[1],patch_diag2[1]])+10] ] 
mx_query[zslice-2:zslice+2, ext_x[0]:ext_x[1], ext_y[0]:ext_y[1]] = 1


# masked wsubvol1 nby probabilities
vol1 = mriVols[1][zslice,:,:]
vol2 = mriVols[2][zslice,:,:]
vol3 = mriVols[3][zslice,:,:]
vol4 = mriVols[4][zslice,:,:]

wvol1 = probmap[zslice,:,:]*vol1
wvol2 = probmap[zslice,:,:]*vol2
wvol3 = probmap[zslice,:,:]*vol3
wvol4 = probmap[zslice,:,:]*vol4

wvols = [wvol1,wvol2,wvol3,wvol4]

fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(15, 20))
for k in range(1,5):
    ax[k-1,0].imshow(mriVols[k][zslice,:,:], cmap=plt.cm.gray)
    ax[k-1,0].set_adjustable('box-forced')
    ax[k-1,0].set_xlabel('zlice'+str(zslice+1)+'_'+str(k)+'mcMRI x')
    
    ax[k-1,1].imshow(probmap[zslice,:,:], cmap=plt.cm.gray)
    ax[k-1,1].axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))
    ax[k-1,1].set_xlabel('lesion probmap = ')
    
    ax[k-1,2].imshow(wvols[k-1], cmap=plt.cm.gray)
    ax[k-1,2].axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))
    ax[k-1,2].set_xlabel('WEIGHTED PROB IMAGE + NODES')
    
# to collect allcoords_wv
allcoords_wv = []
allcoords_wprob = []
for kw,wvol in enumerate(wvols):
    cuts = np.percentile(wvol, 99)
    mx_seg_lesion = ma.masked_array(np.zeros( wvol.shape ), mask=wvol>cuts)
    wv_masked_seg_lesion = ma.filled(mx_seg_lesion, fill_value=1.0)
    # Using the “marching squares” method to compute a the iso-valued contours 
    outlines_probmap = find_contours(wv_masked_seg_lesion, 0)
    coords_probmap = []
    for oi, outline in enumerate(outlines_probmap):
        #cords_redc = approximate_polygon(outline, tolerance=0.05)                
        inside = np.asarray([ext_x[0]<cr[0]<ext_x[1] and ext_y[0]<cr[1]<ext_y[1] for cr in outline]).any()
        if(inside):
            coords_probmap.append( outline )
            # plot
            ax[kw,2].plot(outline[:, 1], outline[:, 0], '.r', linewidth=1)
            
    # perform skeletonization
    wv_skeleton = skeletonize(wv_masked_seg_lesion.astype(bool))       
    # find points in the skeleton
    coords_wv_skeleton = np.column_stack(np.where(wv_skeleton))
    inside = np.asarray([ext_x[0]<cr[0]<ext_x[1] and ext_y[0]<cr[1]<ext_y[1] for cr in coords_wv_skeleton])
    coords_wv_skeleton = coords_wv_skeleton[inside]
    ax[kw,2].plot(coords_wv_skeleton[:, 1], coords_wv_skeleton[:, 0], '.c', linewidth=1)

    # once collected append them all as a 2D array of points
    coords_probmap = np.vstack(([coords for coords in coords_probmap]))
    coords_wv_skeleton = np.vstack(([coords for coords in coords_wv_skeleton]))
    coords_wv = np.vstack((coords_probmap,coords_wv_skeleton))
    # for probabilities
    outline_wprob = [wvol[int(u),int(v)] for u,v in coords_probmap]
    skeleton_wprob = [wvol[int(u),int(v)] for u,v in coords_wv_skeleton]
    allcoords_wprob.append( np.hstack((outline_wprob,skeleton_wprob)) )
    allcoords_wv.append( coords_wv )    


# append cords and Visualize the weighted probability values at the node locations
allcoords_wv = np.vstack(([coords for coords in allcoords_wv]))
allcoords_wprob = np.hstack(([coords for coords in allcoords_wprob]))
prob_img = np.zeros(mriVols[4][zslice,:,:].shape)
c=0
for u,v in allcoords_wv:
    prob_img[int(u),int(v)] = allcoords_wprob[c]
    c+=1

fig, ax = plt.subplots(figsize=(8, 8))
improb = ax.imshow(prob_img.astype('uint8'), cmap=plt.cm.plasma)
ax.set_adjustable('box-forced')
ax.set_xlabel('WEIGHTED PROB map  + allcoords_wprob')
ax.axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))

v = np.linspace(min(prob_img.flatten()), max(prob_img.flatten()), 10, endpoint=True)     
divider = make_axes_locatable(ax)
caxEdges = divider.append_axes("right", size="20%", pad=0.05)
plt.colorbar(improb, cax=caxEdges, ticks=v) 
         
####################
### 1) Extract lesion SI/enhancement
####################    
mask_queryVols = []
onlyROI = []
for k in range(5):
    mx = ma.masked_array(mriVols[k], mask=mx_query==0)
    print "masked lesionVol_%i, lesion mean SI/enhancement = %f" % (k, mx.mean())
    mask_queryVols.append( ma.filled(mx, fill_value=None) )
    onlyROI.append( ma.compress_rowcols( mask_queryVols[k][zslice,:,:] ))

# Compute SI difference
rSIvols = []
for k in range(1,len(onlyROI)):
     rSIvols.append( onlyROI[k] - onlyROI[0] )
     print "lesion rSIvol_s%i, lesion mean realative SI/enhancement = %f" % (k, rSIvols[k-1].mean())

####################
# ROI network formation 
# get list of nodes       
allcoords_wv_redc = approximate_polygon(allcoords_wv, tolerance=0.01)      
allcoords_wv_redc.shape
nodepts = np.asarray( [allcoords_wv_redc[:, 1], allcoords_wv_redc[:, 0]]).transpose()
y = np.ascontiguousarray(nodepts).view(np.dtype((np.void, nodepts.dtype.itemsize * nodepts.shape[1])))
_, idx = np.unique(y, return_index=True)
unique_nodepts = nodepts[idx]
pts = [tuple(pi.flatten()) for pi in unique_nodepts]            


#############################
###### 1) Sample rSI at nodes
#############################
nodew = []
for node in pts:
    loc = tuple([int(loc) for loc in node])
    # find node location accros rSIvols
    rSIt = np.asarray([rSIvol[loc[1],loc[0]] for rSIvol in rSIvols])
    nodew.append( rSIt )

print 'nodeweights a total of {} computed...'.format(len(nodew))

#############################
###### 2) create placeholder for nx nodes
#############################                        
nodes = list(range(len(pts)))
# get original position of points
pos = dict(zip(nodes,pts))
# mapping from vertices to nodes
m = dict(enumerate(nodes)) 

#Create a graph
lesionG = nx.Graph() 
for i in range(len(pts)):
    # add position as node attributes
    lesionG.add_node(i, pos=pos[i])
    
                
#############################
###### 4) Compute node similarity
#############################
D = np.zeros((len(nodes),len(nodes)))
RMSD_matrix = np.zeros((len(nodes),len(nodes)))
mask_edgew = np.triu(np.ones((len(nodew),len(nodew))))
for i in range(len(nodes)):
    for j in range(len(nodes)):
        if(mask_edgew[i,j]==1.0 and i!=j):
            RMSD = np.sqrt( np.sum(np.square(nodew[i] - nodew[j])) )   
            # append to matrix
            RMSD_matrix[i,j] = RMSD

            # calculate distance
            nipos = nx.get_node_attributes(lesionG,'pos')[i]
            njpos = nx.get_node_attributes(lesionG,'pos')[j]
            D_ij = np.sqrt(np.sum(np.square( np.asarray([d for d in nipos]) - np.asarray([d for d in njpos]) )))

            # append to matrix    
            D[i,j] = D_ij   
                
                
####################
# Calculate the energy of linking ij
#~~~~~~~ E({A,B}∈L)= e−D/gf * RMSD
gfs = [0.01,0.1,1,10,50,100,1000,10000]  
 
for gf in gfs:           
    E_ij = np.exp(-D/gf)*1.0/RMSD_matrix
    E_ij = np.triu( E_ij, k= 1 )
     
    #Create a graph
    lesionG = nx.Graph() 
    for i in range(len(pts)):
        # add position as node attributes
        lesionG.add_node(i, pos=pos[i])
    
    #############################
    ###### 5) Create links based on energy 
    #############################
    # add weights by topology but weighted by node similarity
    for i in range(len(nodes)): 
        if(np.max(E_ij[:,i]) != 0):
            highE_nodes = np.where(E_ij[:,i] == np.max(E_ij[:,i])) 
            for nE in highE_nodes[0]:
                lesionG.add_edge( nE, i, weight = E_ij[nE,i])  
    
    # PLOT
    fig, ax = plt.subplots(figsize=(12,12), dpi=160)
    # del graph
    ax.imshow(mriVols[4][zslice,:,:], cmap=plt.cm.gray)
    ax.set_adjustable('box-forced')
    ax.axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))
    nx.draw_networkx(lesionG, pos, ax=ax, with_labels=False, node_color='r',node_size=50, linewidths=None, edge_color='c', width=1.5)
    ax.set_xlabel('4th postc mriVol + lesion graph GF='+str(gf),fontdict=axis_font)                   
                
                
                
#######################################################################################             
import seaborn as sns      

#############################
###### 3) Compute node similarity
#############################
gf = 10
#~~~~~~~ E({A,B}∈L)= e−D/gf * RMSD
D = np.zeros((len(nodes),len(nodes)))
RMSD_matrix = np.zeros((len(nodes),len(nodes)))
E_ij = np.zeros((len(nodes),len(nodes)))
mask_edgew = np.triu(np.ones((len(nodew),len(nodew))))   
for i in range(len(nodes)):            
    for j in range(len(nodes)):
        if(mask_edgew[i,j]==1.0 and i!=j):
            ## the larger the generalized variance the more dispersed the data are
            RMSD = np.sqrt( np.sum(np.square(nodew[i] - nodew[j])) )   
            # append to matrix
            RMSD_matrix[i,j] = RMSD
            
            # calculate distance
            nipos = nx.get_node_attributes(lesionG,'pos')[i]
            njpos = nx.get_node_attributes(lesionG,'pos')[j]
            D_ij = np.sqrt(np.sum(np.square( np.asarray([d for d in nipos]) - np.asarray([d for d in njpos]) )))
   
            # append to matrix    
            D[i,j] = D_ij   
            ####################
            # Calculate the energy of linking ij
            E_ij[i,j] = np.exp(-D[i,j]/gf)*1.0/RMSD_matrix[i,j]                 
 
#############################
###### 4) Create links based on energy 
#############################
# add weights by  topology but weighted by node similarity
##E_ij = np.exp(-RMSD_matrix/gf)*1.0/D
##E_ij = np.triu( E_ij, k= 1 )
for i in range(len(nodes)):
    if(np.max(E_ij[:,i]) != 0):
        highE_nodes = np.where(E_ij[:,i] == np.max(E_ij[:,i])) 
        allinf = sum([E_ij[nE,i]==inf for nE in highE_nodes[0]]) == len(highE_nodes[0])
        if not allinf:
            for nE in highE_nodes[0]:
                if(E_ij[nE,i]==inf):
                    lesionG.add_edge( nE, i, weight = 1.0/gf)  
                else:
                    lesionG.add_edge( nE, i, weight = E_ij[nE,i])  
        else:
            nE = highE_nodes[0][len(highE_nodes[0])-1]
            lesionG.add_edge( nE, i, weight = 1.0/gf)
        
                

############################################################################
## Network connectivity properties
############################################################################

import pandas as pd
import numpy as np
import os, sys
import os.path
import shutil
import glob
import tempfile
import subprocess
import SimpleITK as sitk

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.insert(0,'Z:\\Cristina\Section3\\breast_MR_NME_biological')
from query_localdatabase import *
import glob
import six.moves.cPickle as pickle
import gzip
import networkx as nx

graphs_path = 'Z:\\Cristina\\Section3\\breast_MR_NME_biological\\processed_NMEs'
nxGfeatures_path = 'Z:\\Cristina\\Section3\\breast_MR_NME_biological\\NMEs_SER_nxgmetrics'
mha_data_loc= 'Z:\\Cristina\mha'
processed_path = r'Z:\Cristina\Section3\breast_MR_NME_pipeline\processed_data'


lesion_id = 107
localdata = Querylocaldb()
dflesion  = localdata.querylocalDatabase_wRad(lesion_id)      
cond = dflesion[0]
lesion_record = dflesion[1]
roi_record = dflesion[2]
nmlesion_record = dflesion[3]
StudyID = lesion_record['cad_pt_no_txt']
AccessionN = lesion_record['exam_a_number_txt']
DynSeries_id = nmlesion_record['DynSeries_id']  
roi_id = roi_record['roi_id']
label = roi_record['roi_label']
c = roi_record['roi_centroid']
centroid = c[c.find("(")+1:c.find(")")].split(',')
zslice = int(roi_record['zslice'])
p1 = roi_record['patch_diag1']
patch_diag1 = p1[p1.find("(")+1:p1.find(")")].split(',')
patch_diag1 = [float(p) for p in patch_diag1]
p2 = roi_record['patch_diag2']
patch_diag2 = p2[p2.find("(")+1:p2.find(")")].split(',')
patch_diag2 = [float(p) for p in patch_diag2]    

print("====================")
print('StudyID: ', StudyID)
print('AccessionN: ', AccessionN)
print('DynSeries_id: ', DynSeries_id)
print('label: ', label)
print('lesion_id: ', lesion_id)
print('roi_id: ', roi_id)
print("====================")

#############################
###### 1) Accesing mc images and lesion prob maps
#############################
# get dynmic series info
precontrast_id = int(DynSeries_id) 
DynSeries_nums = [str(n) for n in range(precontrast_id,precontrast_id+5)]

print "Reading MRI volumes..."
DynSeries_imagefiles = []
mriVols = []
preCon_filename = '{}_{}_{}'.format(StudyID.zfill(4),AccessionN,DynSeries_nums[0] )
print preCon_filename
glob_result = glob.glob(os.path.join(processed_path,preCon_filename+'@*')) #'*':do not to know the exactly acquistion time
if glob_result != []:
    filename = glob_result[0]
# read Volumnes
DynSeries_imagefiles.append(filename)
mriVolDICOM = sitk.ReadImage(filename)
mriVols.append( sitk.GetArrayFromImage(sitk.Cast(mriVolDICOM,sitk.sitkFloat32)) )
mriVolSize = mriVolDICOM.GetSize()
print "MRI volumes Size = [%f,%f,%f]..." % mriVolSize
mriVolSpacing = mriVolDICOM.GetSpacing()
print "MRI volumes spacing = [%f,%f,%f]..." % mriVolSpacing
mriVolVoxratio = mriVolSpacing[2]/mriVolSpacing[0]        

ext_x = [int(ex) for ex in [np.min([patch_diag1[0],patch_diag2[0]])-30,np.max([patch_diag1[0],patch_diag2[0]])+25] ] 
ext_y = [int(ey) for ey in [np.min([patch_diag1[1],patch_diag2[1]])-25,np.max([patch_diag1[1],patch_diag2[1]])+40] ] 

for j in range(1,5):
    DynSeries_filename = '{}_{}_{}'.format(StudyID.zfill(4),AccessionN,DynSeries_nums[j] )
    glob_result = glob.glob(os.path.join(processed_path,DynSeries_filename+'@*')) 
    if glob_result != []:
        filename = [name for name in glob_result if '_mc' in name][0] #glob_result[0]
        print filename

    # add side info from the side of the lesion
    DynSeries_imagefiles.append(filename)
    # read Volumnes
    mriVolDICOM = sitk.ReadImage(filename)
    mriVols.append( sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(DynSeries_imagefiles[j]),sitk.sitkFloat32)) )

## to read graph
with gzip.open(os.path.join(graphs_path,'{}_{}_{}_{}_lesion_nxgraph.pklz'.format(roi_id,StudyID.zfill(4),AccessionN,label)), 'rb') as f:
    try:
        lesionG = pickle.load(f, encoding='latin1')
    except:
        lesionG = pickle.load(f)

import seaborn as sns
sns.set_style("darkgrid", {'axes.grid' : False, "legend.frameon": True})
sns.set_context("paper", font_scale=2)  
nodes = [u for (u,v) in lesionG.nodes(data=True)]
pts = [v.values()[0] for (u,v) in lesionG.nodes(data=True)]
pos = dict(zip(nodes,pts))

average_neighbor_degree = nx.average_neighbor_degree(lesionG)
pd_average_neighbor_degree = pd.Series(average_neighbor_degree.values(), name="average_neighbor_degree")
fig, ax = plt.subplots(figsize=(6,6), dpi=160)
ax.hist(average_neighbor_degree.values(),color='b')
sns.distplot(pd_average_neighbor_degree, label="average_neighbor_degree", ax=ax, hist=False)

fig, ax = plt.subplots(figsize=(16,16), dpi=160)
ax.imshow(mriVols[4][zslice,:,:], cmap=plt.cm.gray)
average_neighbor_degreevalues = np.asarray([average_neighbor_degree.get(node) for node in lesionG.nodes()])
v = np.linspace(min(average_neighbor_degreevalues), max(average_neighbor_degreevalues), 10, endpoint=True) 
nxg = nx.draw_networkx_nodes(lesionG, pos, ax=ax, node_color=average_neighbor_degreevalues, cmap=plt.cm.jet,  
                 node_vmin=min(average_neighbor_degreevalues), node_vmax=max(average_neighbor_degreevalues),
                 with_labels=False, node_size=10)
nx.draw_networkx_edges(lesionG, pos, ax=ax,  width=0.5, edge_color='w')
ax.axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))
ax.set_axis_off()
ax.set_adjustable('box-forced')
ax.set_xlabel('average_neighbor_degree')    
divider = make_axes_locatable(ax)
caxEdges = divider.append_axes("right", size="10%", pad=0.05)
plt.colorbar(nxg, cax=caxEdges, ticks=v) 



Betweenness = nx.betweenness_centrality(lesionG)
pd_Betweenness = pd.Series(Betweenness.values(), name="Betweenness")
fig, ax = plt.subplots(figsize=(6,6), dpi=160)
ax.hist(Betweenness.values(),color='b')
sns.distplot(pd_Betweenness, label="Betweenness", ax=ax, hist=False)

fig, ax = plt.subplots(figsize=(16,16), dpi=160)
ax.imshow(mriVols[4][zslice,:,:], cmap=plt.cm.gray)
Betweennessvalues = np.asarray([Betweenness.get(node) for node in lesionG.nodes()])
v = np.linspace(min(Betweennessvalues), max(Betweennessvalues), 10, endpoint=True) 
nxg = nx.draw_networkx_nodes(lesionG, pos, ax=ax, node_color=Betweennessvalues, cmap=plt.cm.jet,  
                 node_vmin=min(Betweennessvalues), node_vmax=max(Betweennessvalues),
                 with_labels=False, node_size=10)
nx.draw_networkx_edges(lesionG, pos, ax=ax, width=0.5, edge_color='w')
ax.axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))
ax.set_axis_off()
ax.set_adjustable('box-forced')
ax.set_xlabel('Degree nodes')    
divider = make_axes_locatable(ax)
caxEdges = divider.append_axes("right", size="10%", pad=0.05)
plt.colorbar(nxg, cax=caxEdges, ticks=v) 



closeness = nx.closeness_centrality(lesionG)
pd_closeness = pd.Series(closeness.values(), name="closeness")
fig, ax = plt.subplots(figsize=(6,6), dpi=160)
ax.hist(closeness.values(),color='b')
sns.distplot(pd_closeness, label="closeness", ax=ax, hist=False)

fig, ax = plt.subplots(figsize=(16,16), dpi=160)             
ax.imshow(mriVols[4][zslice,:,:], cmap=plt.cm.gray)
Closenvalues = np.asarray([closeness.get(node) for node in lesionG.nodes()])
v = np.linspace(min(Closenvalues), max(Closenvalues), 10, endpoint=True) 
nxg = nx.draw_networkx_nodes(lesionG, pos, ax=ax, node_color=Closenvalues, cmap=plt.cm.jet,  
                 node_vmin=min(Closenvalues), node_vmax=max(Closenvalues),
                 with_labels=False, node_size=10)
nx.draw_networkx_edges(lesionG, pos, ax=ax, width=0.5, edge_color='w')
ax.axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))
ax.set_axis_off()
ax.set_adjustable('box-forced')
ax.set_xlabel('Closeness')    
divider = make_axes_locatable(ax)
caxEdges = divider.append_axes("right", size="10%", pad=0.05)
plt.colorbar(nxg, cax=caxEdges, ticks=v) 


## NOTE: clustering coefficient replaced/updated as follows:
# recalculate clustering coefficient
import itertools
nodes = [u for (u,v) in lesionG.nodes(data=True)]
#weights = [d['weight'] for (u,v,d) in lesionG.edges(data=True)]
edgesdict = lesionG.edge
clustering = []
for nij  in nodes:
    #print 'node = %d' % nij
    # The pairs must be given as 2-tuples (u, v) where u and v are nodes in the graph. 
    node_adjacency_dict = edgesdict[nij]
    cc_total = 0.0
    for pairs in itertools.combinations(lesionG.neighbors_iter(nij), 2):
        #print pairs
        adjw = np.sum([node_adjacency_dict[pairs[0]].get("weight",0), node_adjacency_dict[pairs[1]].get("weight",0)])/2.0
        cc_total += adjw
    #print cc_total    
    clustering.append( cc_total )
    
pd_clustering = pd.Series(clustering, name="clustering")
fig, ax = plt.subplots(figsize=(6,6), dpi=160)
ax.hist(clustering,color='b')
sns.distplot(pd_clustering, label="clustering", ax=ax, hist=False)

fig, ax = plt.subplots(figsize=(16,16), dpi=160)             
ax.imshow(mriVols[4][zslice,:,:], cmap=plt.cm.gray)
clustering_vals = np.asarray([c for c in clustering])
v = np.linspace(min(clustering_vals), max(clustering_vals), 10, endpoint=True) 
nxg = nx.draw_networkx_nodes(lesionG, pos, ax=ax, node_color=clustering_vals, cmap=plt.cm.jet,  
                 node_vmin=min(clustering_vals), node_vmax=max(clustering_vals),
                 with_labels=False, node_size=10)
nx.draw_networkx_edges(lesionG, pos, ax=ax, width=0.5, edge_color='w')
ax.axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))
ax.set_axis_off()
ax.set_adjustable('box-forced')
ax.set_xlabel('clustering')    
divider = make_axes_locatable(ax)
caxEdges = divider.append_axes("right", size="10%", pad=0.05)
plt.colorbar(nxg, cax=caxEdges, ticks=v)    
