import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
import os
import math
from scipy.spatial import cKDTree

import scSLAT
from scSLAT.model import Cal_Spatial_Net, load_anndatas, run_SLAT, spatial_match
from scSLAT.viz import match_3D_multi, hist, Sankey
from scSLAT.metrics import region_statistics


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



path1 = '../../../data/starmap/13month_cr1.h5ad'
path2 = '../../../data/starmap/13month_cr2.h5ad'
angles = [45,90,135,180,225,270]

fig, axs = plt.subplots(6, 1)

df = pd.DataFrame(columns=['pcc','cri','pred_angle','real_angle'])
for i in range(6):
#i=0
    adata1 = sc.read_h5ad(path1)
    adata2 = sc.read_h5ad(path2)
    adata1.obsm['spatial'] = np.array(adata1.obsm['spatial'])
    adata2.obsm['spatial'] = np.array(adata2.obsm['spatial'])
    adata1.obsm['spatial'][:, 0] = adata1.obsm['spatial'][:, 0] - np.mean(adata1.obsm['spatial'][:, 0])
    adata1.obsm['spatial'][:, 1] = adata1.obsm['spatial'][:, 1] - np.mean(adata1.obsm['spatial'][:, 1])
    adata2.obsm['spatial'][:, 0] = adata2.obsm['spatial'][:, 0] - np.mean(adata2.obsm['spatial'][:, 0])
    adata2.obsm['spatial'][:, 1] = adata2.obsm['spatial'][:, 1] - np.mean(adata2.obsm['spatial'][:, 1])
    angle = angles[i]
    radian = math.radians(angle)
    rotation = np.array([[np.cos(radian), -np.sin(radian)],
                        [np.sin(radian), np.cos(radian)]])
    adata2.obsm['spatial'] = np.dot(adata2.obsm['spatial'], rotation.T)

    Cal_Spatial_Net(adata1, k_cutoff=3, model='KNN')
    Cal_Spatial_Net(adata2, k_cutoff=3, model='KNN')
    edges, features = load_anndatas([adata1, adata2], feature='raw',check_order=False)
    embd0, embd1, time = run_SLAT(features, edges)
    best, index, distance = spatial_match(features, adatas=[adata1,adata2], reorder=False)
    matching = np.array([range(index.shape[0]), best])
    R,T = find_rigid_transform(adata1.obsm['spatial'][matching[1,:]],adata2.obsm['spatial'])
    pred_angle = rotation_angle_2d(R)
    adata1.obsm['align_spatial'] = np.dot(adata1.obsm['spatial'], rotation.T) + T
    pcc,cri = evaluation(adata1.obsm['align_spatial'],
                           adata2.obsm['spatial'],
                           adata1.X, 
                           adata2.X, 
                           adata1.obsm['celltype'], 
                           adata2.obsm['celltype'])
    df = df._append(pd.Series({'pcc': pcc, 'cri': cri,'pred_angle':pred_angle,'real_angle':angle}),ignore_index=True)
    plt.figure(figsize=(4,4))
    plt.scatter(adata2.obsm['spatial'][:,0],adata2.obsm['spatial'][:,1],c='b',label='target',alpha=0.2,s=0.8)
    plt.scatter(adata1.obsm['align_spatial'][:,0],adata1.obsm['align_spatial'][:,1],c='r',label='source',alpha=0.2,s=0.8)
    plt.axis('off')
    plt.savefig(f'../../results/Starmap/figs/13_SLAT_rotation_{angle}.png',dpi=300)
df.to_csv('../../results/Starmap/SLAT_13.csv')