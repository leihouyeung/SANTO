import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

import os
import math
import torch
from scipy.spatial import cKDTree
from train_STAligner import train_STAligner
from ST_utils import Cal_Spatial_Net
used_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
used_device = 'cpu'
print(used_device)

def find_rigid_transform(A, B):
    assert A.shape == B.shape

    # 计算中心点
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # 中心化点集
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # 计算协方差矩阵
    H = A_centered.T @ B_centered

    # 使用SVD计算旋转矩阵
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 计算位移向量
    t = -R @ centroid_A + centroid_B

    return R, t

def rotation_angle_2d(R):
    theta = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees(theta)  # 将弧度转换为度
def evaluation(src_cor, tgt_cor, src_exp, tgt_exp, src_cell_type, tgt_cell_type):

    kd_tree = cKDTree(src_cor)
    distances, indices = kd_tree.query(tgt_cor, k=1) 
    corr = np.corrcoef(np.concatenate((tgt_exp,src_exp[indices]), axis=0))[:tgt_exp.shape[0],tgt_exp.shape[0]:]
    acc = corr.trace()/tgt_exp.shape[0]
    cri = np.mean((tgt_cell_type == src_cell_type[indices])+0)
    #euc = np.mean((ori_src_cor-src_cor)**2)
    
    return acc, cri

adata = sc.read_h5ad('../../data/MERFISH/12_slices_1.h5ad')
#adata.obsm['align_spatial'] = adata.obsm['spatial']
slice_ids = np.unique(adata.obs.Bregma)
slice_ids[::-1].sort()
adatas = [adata[adata.obs.Bregma == i,:].copy() for i in slice_ids]
section_ids = slice_ids
Batch_list = []
adj_list = []
df = pd.DataFrame(columns=['pcc','cri'])

for adata in adatas:    
    Cal_Spatial_Net(adata, rad_cutoff=150) # the spatial network are saved in adata.uns[‘adj’]
    # STAligner.Stats_Spatial_Net(adata) # plot the number of spatial neighbors

    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=50)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']].copy()

    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)
    
adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('str')

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(section_ids)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)
adata_concat = train_STAligner(adata_concat, verbose=True, knn_neigh = 10, device=used_device)

for i in range(len(slice_ids)-1):
    adata1 = adata_concat[adata_concat.obs.slice_name == slice_ids[i],:].copy()
    adata2 = adata_concat[adata_concat.obs.slice_name == slice_ids[i+1],:].copy()

    kd_tree = cKDTree(adata1.obsm['STAligner'])
    distances, indices = kd_tree.query(adata2.obsm['STAligner'], k=1) 
    R,T = find_rigid_transform(adata1.obsm['spatial'][indices],adata2.obsm['spatial'])
    pred_angle = rotation_angle_2d(R)
    adata_concat.obsm['spatial'][adata_concat.obs.Bregma >= slice_ids[i],:] = np.dot(adata_concat.obsm['spatial'][adata_concat.obs.Bregma >= slice_ids[i],:], R.T) + T
    # pcc,cri = evaluation(adata_concat.obsm['spatial'][adata_concat.obs.Bregma == slice_ids[i],:],
    #                        adata2.obsm['spatial'],
    #                        adata1.X, 
    #                        adata2.X, 
    #                        np.array(adata1.obs['Cell_class']), 
    #                        np.array(adata2.obs['Cell_class']))
    #df = df.append(pd.Series({'pcc': pcc, 'cri': cri}),ignore_index=True)
adata_concat.write_h5ad('./STAligner_1.h5ad')
# df.to_csv('../../results/MERFISH/STAligner_ID1.csv')
# sc.pl.spatial(adata_concat,
#               basis = 'spatial',
#               color='Cell_class',
#               spot_size=20,
#               save='MERFISH_STAligner_ID1_2D.svg')


adata = sc.read_h5ad('../../data/MERFISH/12_slices_2.h5ad')
#adata.obsm['align_spatial'] = adata.obsm['spatial']
slice_ids = np.unique(adata.obs.Bregma)
slice_ids[::-1].sort()
adatas = [adata[adata.obs.Bregma == i,:].copy() for i in slice_ids]
section_ids = slice_ids
Batch_list = []
adj_list = []
df = pd.DataFrame(columns=['pcc','cri'])

for adata in adatas:    
    Cal_Spatial_Net(adata, rad_cutoff=150) # the spatial network are saved in adata.uns[‘adj’]
    # STAligner.Stats_Spatial_Net(adata) # plot the number of spatial neighbors

    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=50)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']].copy()

    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)
    
adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('str')

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(section_ids)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)
adata_concat = train_STAligner(adata_concat, verbose=True, knn_neigh = 10, device=used_device)

for i in range(len(slice_ids)-1):
    adata1 = adata_concat[adata_concat.obs.slice_name == slice_ids[i],:].copy()
    adata2 = adata_concat[adata_concat.obs.slice_name == slice_ids[i+1],:].copy()

    kd_tree = cKDTree(adata1.obsm['STAligner'])
    distances, indices = kd_tree.query(adata2.obsm['STAligner'], k=1) 
    R,T = find_rigid_transform(adata1.obsm['spatial'][indices],adata2.obsm['spatial'])
    pred_angle = rotation_angle_2d(R)
    adata_concat.obsm['spatial'][adata_concat.obs.Bregma >= slice_ids[i],:] = np.dot(adata_concat.obsm['spatial'][adata_concat.obs.Bregma >= slice_ids[i],:], R.T) + T
#     pcc,cri = evaluation(adata_concat.obsm['spatial'][adata_concat.obs.Bregma == slice_ids[i],:],
#                            adata2.obsm['spatial'],
#                            adata1.X, 
#                            adata2.X, 
#                            np.array(adata1.obs['Cell_class']), 
#                            np.array(adata2.obs['Cell_class']))
#     df = df.append(pd.Series({'pcc': pcc, 'cri': cri}),ignore_index=True)
# df.to_csv('../../results/MERFISH/STAligner_ID2.csv')
# sc.pl.spatial(adata_concat,
#               basis = 'spatial',
#               color='Cell_class',
#               spot_size=20,
#               save='MERFISH_STAligner_ID2_2D.svg')

adata_concat.write_h5ad('./STAligner_1.h5ad')