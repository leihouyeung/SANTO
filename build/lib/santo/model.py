#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def nearest_neighbor(src, dst):
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,dim=0,keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=3):
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    return feature



class DGCNN_cor(nn.Module):
    def __init__(self, cor_dim = 2, k =3):
        super(DGCNN_cor, self).__init__()
        self.k = k
        self.conv1 = nn.Conv2d(cor_dim*2, 32, kernel_size=1, bias=False) # input: 2*2 or 3*2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(32+64+128+256, 512, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x,self.k)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]
        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]
        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]
        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x

class DGCNN_exp(nn.Module):
    def __init__(self, exp_dim = 1000, k=3):
        super(DGCNN_exp, self).__init__()
        self.k = k
        self.conv1 = nn.Conv2d(2*exp_dim, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(64+128+256+512, 1024, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x,self.k)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]
        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]
        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]
        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x



class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.dimension = args.dimension
        if args.dimension == 2:
            self.reflect = nn.Parameter(torch.eye(2), requires_grad=False)
            self.reflect[1, 1] = -1
        if args.dimension == 3:
            self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
            self.reflect[2, 2] = -1

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size = src.size(0)

        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2) # (1,src,tgt)
        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())
        src_centered = src - src.mean(dim=2, keepdim=True)
        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)
        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                # r = r * self.reflect
            R.append(r)
            U.append(u)
            S.append(s)
            V.append(v)

        R = torch.stack(R, dim=0)
        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, self.dimension)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.exp_dim = args.exp_dim
        self.dimension = args.dimension
        self.k = args.k
        self.cor_emb_nn = DGCNN_cor(cor_dim = self.dimension,k=args.k)
        self.exp_emb_nn = DGCNN_exp(exp_dim = self.exp_dim,k=args.k)
        self.head = SVDHead(args=args)

    def forward(self, *input):
        src_cor = input[0].transpose(2,1).contiguous() # (Batch_size, Feature, num_point)
        src_exp = input[1].transpose(2,1).contiguous() # (Batch_size, Feature, num_point)
        tgt_cor = input[2].transpose(2,1).contiguous() # (Batch_size, Feature, num_point)
        tgt_exp = input[3].transpose(2,1).contiguous() # (Batch_size, Feature, num_point)

        src_cor_embedding = self.cor_emb_nn(src_cor) # (Batch_size, cor_emb_feature, num_point)
        src_exp_embedding = self.exp_emb_nn(src_exp) # (Batch_size, exp_emb_feature, num_point)
        tgt_cor_embedding = self.cor_emb_nn(tgt_cor) # (Batch_size, cor_emb_feature, num_point)
        tgt_exp_embedding = self.exp_emb_nn(tgt_exp) # (Batch_size, exp_emb_feature, num_point)

        src_embedding = torch.cat((src_cor_embedding,src_exp_embedding),dim=1)
        tgt_embedding = torch.cat((tgt_cor_embedding,tgt_exp_embedding),dim=1)

        rotation_ab, translation_ab = self.head(src_embedding, tgt_embedding, src_cor, tgt_cor)

        return rotation_ab, translation_ab
