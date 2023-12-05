
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad


def intersect(lst1, lst2):
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def combine_training_data(src, tgt, is_preprocess=True):
    if not is_preprocess:
        sc.pp.normalize_total(src)
        sc.pp.log1p(src)
        sc.pp.normalize_total(tgt)
        sc.pp.log1p(tgt)

    # centralize the coordinates to (0,0)
    src_coo = np.array(src.obsm['spatial']).astype(np.float32)  # (num_points, num_dimensions)
    tgt_coo = np.array(tgt.obsm['spatial']).astype(np.float32)


    scale1 = np.max(src_coo[:, 0]) - np.min(src_coo[:, 0])
    scale2 = np.max(src_coo[:, 1]) - np.min(src_coo[:, 1])

    src_coo[:, 0] = (src_coo[:, 0] - np.min(src_coo[:, 0])) / scale1
    src_coo[:, 1] = (src_coo[:, 1] - np.min(src_coo[:, 1])) / scale2
    if src_coo.shape[1] ==3:
        scale3 = np.max(src_coo[:, 2]) - np.min(src_coo[:, 2])
        src_coo[:, 2] = (src_coo[:, 2] - np.min(src_coo[:, 2])) / scale3
        src_coo[:, 2] = src_coo[:, 2] - np.mean(src_coo[:, 2])


    tgt_coo[:, 0] = (tgt_coo[:, 0] - np.min(tgt_coo[:, 0])) / scale1
    tgt_coo[:, 1] = (tgt_coo[:, 1] - np.min(tgt_coo[:, 1])) / scale2


    if tgt_coo.shape[1] ==3:
        tgt_coo[:, 2] = (tgt_coo[:, 2] - np.min(tgt_coo[:, 2])) / scale3
        tgt_coo[:, 2] = tgt_coo[:, 2] - np.mean(tgt_coo[:, 2])

    var_names = tgt.var.index.to_frame()
    var_names.columns = ['name']
    com_adata = ad.AnnData(
        X=tgt.X,
        var=var_names,
        dtype='float32'
    )
    com_adata.uns['src_exp'] = src.X
    if tgt_coo.shape[1] == 2:
        com_adata.uns['target_spatial'] = pd.DataFrame(tgt_coo, columns=['x', 'y'])
        com_adata.uns['source_spatial'] = pd.DataFrame(src_coo, columns=['x', 'y']) ### the coordinates after coarse alignment
    if tgt_coo.shape[1] == 3:
        com_adata.uns['target_spatial'] = pd.DataFrame(tgt_coo, columns=['x', 'y', 'z'])
        com_adata.uns['source_spatial'] = pd.DataFrame(src_coo, columns=['x', 'y', 'z']) ### the coordinates after coarse alignment

    com_adata.uns['scale1'] = scale1
    com_adata.uns['scale2'] = scale2

    return com_adata


class STDataset(Dataset):
    def __init__(self, adatas):
        self.src_coo = np.array([ad.uns['source_spatial'].to_numpy() for ad in adatas])
        self.src_exp = np.array([ad.uns['src_exp'] for ad in adatas])
        self.tgt_coo = {}
        self.tgt_exp = {}
        self.tgt_coo = np.array([ad.uns['target_spatial'].to_numpy() for ad in adatas])
        self.tgt_exp = np.array([ad.X for ad in adatas])

    def __getitem__(self, item):
        src_coo = self.src_coo[item]
        src_exp = self.src_exp[item]
        tgt_coo = self.tgt_coo[item]
        tgt_exp = self.tgt_exp[item]

        return src_coo.astype('float32'), src_exp.astype('float32'), tgt_coo.astype('float32'), tgt_exp.astype(
                'float32')

    def __len__(self):
        return self.src_coo.shape[0]
