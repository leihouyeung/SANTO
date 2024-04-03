from __future__ import print_function
import torch
import numpy as np
from scipy.sparse import issparse
import pandas as pd
import anndata as ad
import ruptures as rpt
from scipy.spatial import cKDTree
import gc
import scanpy as sc
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from model import Model
from data import intersect, combine_training_data, STDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


def evaluation(src_cor, tgt_cor, src_exp, tgt_exp, src_cell_type, tgt_cell_type):

    kd_tree = cKDTree(src_cor)
    distances, indices = kd_tree.query(tgt_cor, k=1)
    corr = np.corrcoef(np.concatenate((tgt_exp, src_exp[indices]), axis=0))[:tgt_exp.shape[0], tgt_exp.shape[0]:]
    acc = corr.trace() / tgt_exp.shape[0]
    cri = np.mean((tgt_cell_type == src_cell_type[indices]) + 0)

    return acc, cri

def coarse_align(src_cor, tgt_cor, src_exp, tgt_exp, k_list = []):
    if issparse(src_exp):
        src_exp = src_exp.todense()
    if issparse(tgt_exp):
        tgt_exp = tgt_exp.todense()

    if len(k_list) != 0 :
        # process source slice
        knn_src_exp = src_exp.copy()
        kd_tree = cKDTree(src_cor)
        for k in k_list:
            distances, indices = kd_tree.query(src_cor, k=k)  # (source_num_points, k)
            src_exp = src_exp + np.array(np.mean(knn_src_exp[indices, :], axis=1))

        # process target slice
        knn_tgt_exp = tgt_exp.copy()
        kd_tree = cKDTree(tgt_cor)
        for k in k_list:
            distances, indices = kd_tree.query(tgt_cor, k=k)  # (source_num_points, k)
            tgt_exp = tgt_exp + np.array(np.mean(knn_tgt_exp[indices, :], axis=1))

    corr = np.corrcoef(src_exp, tgt_exp)[:src_exp.shape[0],src_exp.shape[0]:]  # (src_points, tgt_points)
    matched_src_cor = src_cor[np.argmax(corr, axis=0), :]

    # Calculate transformation: translation and rotation
    mean_source = np.mean(matched_src_cor, axis=0)
    mean_target = np.mean(tgt_cor, axis=0)
    centered_source = matched_src_cor - mean_source
    centered_target = tgt_cor - mean_target
    rotation_matrix = np.dot(centered_source.T, centered_target)
    u, _, vt = np.linalg.svd(rotation_matrix)
    rotation = np.dot(vt.T, u.T)
    translation = mean_target - np.dot(rotation, mean_source)
    transformed_points = np.dot(src_cor, rotation.T) + translation

    return transformed_points, rotation, translation

def coarse_stitch(src_cor, tgt_cor):

    # Calculate transformation: translation and rotation
    mean_source = np.mean(src_cor, axis=0)
    mean_target = np.mean(tgt_cor, axis=0)
    centered_source = src_cor - mean_source
    centered_target = tgt_cor - mean_target
    rotation_matrix = np.dot(centered_source.T, centered_target)
    u, _, vt = np.linalg.svd(rotation_matrix)
    rotation = np.dot(vt.T, u.T)
    translation = mean_target - np.dot(rotation, mean_source)

    # Apply transformation to source points
    src_cor = np.dot(src_cor, rotation.T) + translation

    return src_cor, np.array(rotation), np.array(translation)

def find_best_matching(src, tgt, k_list=[3, 10, 40]):

    kd_tree = cKDTree(src.obsm['spatial'])
    knn_src_exp_base = src.X.copy()
    knn_src_exp = src.X.copy()
    if issparse(knn_src_exp_base):
        knn_src_exp_base = knn_src_exp_base.todense()
    if issparse(knn_src_exp):
        knn_src_exp = knn_src_exp.todense()
    if len(k_list) != 0:
        for k in k_list:
            distances, indices = kd_tree.query(src.obsm['spatial'], k=k)  # (source_num_points, k)
            knn_src_exp = knn_src_exp + np.array(np.mean(knn_src_exp_base[indices, :], axis=1))

    kd_tree = cKDTree(tgt.obsm['spatial'])
    knn_tgt_exp = tgt.X.copy()
    knn_tgt_exp_base = tgt.X.copy()
    if issparse(knn_tgt_exp_base):
        knn_tgt_exp_base = knn_tgt_exp_base.todense()
    if issparse(knn_tgt_exp):
        knn_tgt_exp = knn_tgt_exp.todense()
    if len(k_list) != 0:
        for k in k_list:
            distances, indices = kd_tree.query(tgt.obsm['spatial'], k=k)  # (source_num_points, k)
            knn_tgt_exp = knn_tgt_exp + np.array(np.mean(knn_tgt_exp_base[indices, :], axis=1))

    corr = np.corrcoef(knn_src_exp, knn_tgt_exp)[:knn_src_exp.shape[0],
           knn_src_exp.shape[0]:]  # (src_points, tgt_points)

    src.X = knn_src_exp
    tgt.X = knn_tgt_exp

    ''' find the spots which are possibly in the overlap region by L1 changepoint detection '''
    y = np.sort(np.max(corr, axis=0))[::-1]
    data = np.array(y).reshape(-1, 1)
    algo = rpt.Dynp(model="l1").fit(data)
    result = algo.predict(n_bkps=1)
    first_inflection_point = result[0]

    ### set1: For each of point in tgt, the corresponding best matched point in src
    set1 = np.array([[index, value]for index, value in enumerate(np.argmax(corr, axis=0))])
    set1 = np.column_stack((set1,np.max(corr, axis=0)))
    set1 = pd.DataFrame(set1,columns = ['tgt_index','src_index','corr'])
    set1.sort_values(by='corr',ascending=False,inplace=True)
    set1 = set1.iloc[:first_inflection_point,:]


    y = np.sort(np.max(corr, axis=1))[::-1]
    data = np.array(y).reshape(-1, 1)
    algo = rpt.Dynp(model="l1").fit(data)
    result = algo.predict(n_bkps=1)
    first_inflection_point = result[0]

    ### set2: For each of point in src, the corresponding best matched point in tgt
    set2 = np.array([[index, value]for index, value in enumerate(np.argmax(corr, axis=1))])
    set2 = np.column_stack((set2,np.max(corr, axis=1)))
    set2 = pd.DataFrame(set2,columns = ['src_index','tgt_index','corr'])
    set2.sort_values(by='corr',ascending=False,inplace=True)
    set2 = set2.iloc[:first_inflection_point,:]


    result = pd.merge(set1, set2, left_on=['tgt_index', 'src_index'], right_on=['tgt_index', 'src_index'], how='inner')
    src_sub = src[result['src_index'].to_numpy().astype(int), :]
    tgt_sub = tgt[result['tgt_index'].to_numpy().astype(int), :]

    return src_sub, tgt_sub,result


def bin_adata(adata, bin_size=1, coords_key='spatial'):
    adata = adata.copy()
    adata.obsm[coords_key] = np.array(adata.obsm[coords_key], dtype=np.float32)
    adata.obsm[coords_key] = (adata.obsm[coords_key] // bin_size).astype(np.int32)

    if issparse(adata.X):
        df = pd.DataFrame(adata.X.A, columns=adata.var_names)
    else:
        df = pd.DataFrame(adata.X, columns=adata.var_names)

    df[["x", "y"]] = np.array(adata.obsm[coords_key])
    df2 = df.groupby(by=["x", "y"]).sum()

    adata_binned = ad.AnnData(df2)
    adata_binned.uns["__type"] = "UMI"
    adata_binned.obs_names = [str(i[0]) + "_" + str(i[1]) for i in df2.index.to_list()]
    adata_binned.obsm[coords_key] = np.array([list(i) for i in df2.index.to_list()], dtype=np.int32)

    return adata_binned


def transform_point_cloud(point_cloud, rotation, translation):
    return torch.matmul(point_cloud,rotation) + translation

def simulate_stitching(adata,axis = 0, from_low = True, threshold = 0.5):
    cadata = adata.copy()
    coo = cadata.obsm['spatial']
    scale = np.max(coo[:, axis]) - np.min(coo[:, axis])
    if from_low:
        chosen_indices = coo[:,axis] > (scale * threshold + np.min(coo[:, axis]))
    else:
        chosen_indices = coo[:,axis] < (np.max(coo[:, axis]) - scale * threshold)
    cadata = cadata[chosen_indices,:].copy()
    return cadata

def custom_softmin(logits, temperature, dim=0):
    exp_logits = torch.exp(-logits * temperature)
    softmax_denominator = torch.sum(exp_logits, dim=dim, keepdim=True)
    probabilities = exp_logits / softmax_denominator
    return probabilities


def train_one_epoch(args, net, train_loader, opt):
    net.train()
    total_loss = 0
    num_examples = 0
    rotations_ab_pred = []
    translations_ab_pred = []

    for src, src_exp, target, target_exp in train_loader:
        src = src.to(args.device)
        src_exp = src_exp.to(args.device)
        target = target.to(args.device)
        target_exp = target_exp.to(args.device)
        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size
        rotation_ab_pred, translation_ab_pred= net(src, src_exp, target, target_exp)

        ## save rotation and translation

        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)
        corr = torch.corrcoef(torch.cat((src_exp[0, :, :], target_exp[0, :, :]), dim=0))[:src_exp.shape[1],
               src_exp.shape[1]:]  # (src_cell_size,tgt_cell_size)
        dis = torch.cdist(transformed_src[0, :, :], target[0, :, :])  # (src,tgt)
        softmin = custom_softmin(dis, 10, dim=1)  # (src,tgt)

        loss = args.alpha * 1 / (src_exp.shape[1]) * ((1 - corr) * softmin).sum() + \
               (1 - args.alpha) * 1 / (src_exp.shape[1]) * (dis * softmin).sum()

        loss.backward()
        opt.step()
        total_loss += loss.item() * batch_size

    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    return total_loss * 1.0 / num_examples, rotations_ab_pred, translations_ab_pred


def train(args, net, train_loader):

    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[30, 60, 90], gamma=0.5)
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:

        scheduler.step()
        train_loss, train_rotations_ab_pred, train_translations_ab_pred = train_one_epoch(args, net, train_loader, opt)
        pbar.set_description(f'Epoch {epoch}, Loss: {train_loss}')
        gc.collect()


def test(net, test_loader, args):
    net.eval()
    rotations_ab_pred = []
    translations_ab_pred = []

    for src, src_exp, target, target_exp in test_loader:
        src = src.to(args.device)
        src_exp = src_exp.to(args.device)
        target = target.to(args.device)
        target_exp = target_exp.to(args.device)
        rotation_ab_pred, translation_ab_pred = net(src, src_exp, target, target_exp)
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
    return rotations_ab_pred, translations_ab_pred

def unify_feature(src, tgt, max_iter = 20):
    '''use harmony to unify the feature space'''
    import harmonypy
    combined_data = ad.concat([src, tgt], join='inner')
    combined_data.obs['ids'] = ['source'] * src.shape[0] + ['target'] * tgt.shape[0]
    sc.tl.pca(combined_data,n_comps=40)
    harmony_out = harmonypy.run_harmony(combined_data.obsm['X_pca'], combined_data.obs, 'ids', max_iter_harmony=max_iter)
    adjusted_matrix = harmony_out.Z_corr.T
    combined_data.obsm['X_pca_harmony'] = adjusted_matrix
    src = combined_data[:src.shape[0],:].copy()
    tgt = combined_data[src.shape[0]:,:].copy()
    new_src = ad.AnnData(X=src.obsm['X_pca_harmony'], obs=src.obs, obsm=src.obsm)
    new_tgt = ad.AnnData(X=tgt.obsm['X_pca_harmony'], obs=tgt.obs, obsm=tgt.obsm)

    return new_src, new_tgt

def santo(src, tgt, args):
    '''

    :param src: source slice
    :param tgt: target slice
    :param args: parameter settings including learning rate, epochs, k, alpha, dimension
    :return: aligned_src_cor: the coordinates after fine alignment
             trans_dict: the transformation dictionary including coarse and fine alignment
    '''

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.cuda.empty_cache()

    comm_gene = intersect(src.var.index, tgt.var.index)
    src = src[:, comm_gene].copy()
    tgt = tgt[:, comm_gene].copy()
    args.exp_dim = len(comm_gene)


    if args.diff_omics:
        src, tgt = unify_feature(src, tgt)
        args.exp_dim = src.shape[1]
    src_cor = np.array(src.obsm['spatial'])
    tgt_cor = np.array(tgt.obsm['spatial'])
    if issparse(src.X):
        src.X = src.X.todense()
    if issparse(tgt.X):
        tgt.X = tgt.X.todense()


    if args.mode == 'align':
        print(f'You choose {args.mode}')
        aligned_source, coarse_R_ab, coarse_T_ab = coarse_align(src_cor, tgt_cor, src.X, tgt.X)
        src.obsm['spatial'] = aligned_source
        src.uns['original_spatial'] = src_cor
        plt.scatter(aligned_source[:, 0], aligned_source[:, 1], c='r', label='transformed', alpha=0.2,s=0.8)

    elif args.mode == 'stitch':
        print(f'You choose {args.mode}')
        src, tgt, result= find_best_matching(src, tgt, k_list=[5,10,20])

        aligned_source, coarse_R_ab, coarse_T_ab = coarse_stitch(src.obsm['spatial'], tgt.obsm['spatial'])
        src.obsm['spatial'] = aligned_source
        src.uns['original_spatial'] = src_cor
        whole_aligned_source = np.dot(src_cor, coarse_R_ab.T) + coarse_T_ab
        plt.scatter(whole_aligned_source[:, 0], whole_aligned_source[:, 1], c='r', label='transformed', alpha=0.2,s=0.8)
        print(f'Coarse stitching is finished, there are {src.shape[0]} matched pairs ')

    else:
        src.uns['original_spatial'] = src_cor
        print('No coarse alignment.')

    ''' Visulization of coarse alignment'''
    plt.scatter(tgt_cor[:, 0], tgt_cor[:, 1], c='b', label='target', alpha=0.2,s=0.8)
    plt.scatter(src_cor[:, 0], src_cor[:, 1], c='g', label='source', alpha=0.2,s=0.8)

    plt.legend()
    plt.title("Visualization of coarse alignment")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

    pair_adatas = combine_training_data(src, tgt, is_preprocess=True)
    scale_factor1 = pair_adatas.uns['scale1']
    scale_factor2 = pair_adatas.uns['scale2']

    data_loader = DataLoader(STDataset([pair_adatas]))
    net = Model(args).to(args.device)

    train(args, net, data_loader)
    rotations_ab_pred, translations_ab_pred = test(net, data_loader, args)

    if rotations_ab_pred[0].shape == (2,2):
        fine_R_ab = rotations_ab_pred[0]
    else:
        fine_R_ab = rotations_ab_pred[0][0,:,:]

    fine_T_ab = translations_ab_pred[0]
    trans_dict = {}
    if args.mode == 'align' or args.mode == 'stitch':
        aligned_src_cor = np.dot(src_cor, coarse_R_ab.T) + coarse_T_ab
        trans_dict['coarse_R_ab'] = coarse_R_ab
        trans_dict['coarse_T_ab'] = coarse_T_ab
    else:
        aligned_src_cor = src_cor

    trans_dict['fine_R_ab'] = fine_R_ab
    trans_dict['fine_T_ab']  = np.array([scale_factor1,scale_factor2]) * fine_T_ab

    ''' Visulization of fine alignment'''
    aligned_src_cor = np.dot(aligned_src_cor, fine_R_ab.T) + np.array([scale_factor1,scale_factor2]) * fine_T_ab
    plt.scatter(tgt_cor[:, 0], tgt_cor[:, 1], c='b', label='target', alpha=0.2,s=0.8)
    plt.scatter(src_cor[:, 0], src_cor[:, 1], c='g', label='source', alpha=0.2,s=0.8)
    plt.scatter(aligned_src_cor[:, 0], aligned_src_cor[:, 1], c='r', label='transformed', alpha=0.2,s=0.8)
    plt.legend()
    plt.title("Visualization of fine alignment")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

    return aligned_src_cor, trans_dict

