import numpy as np
import torch
import torch.sparse
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class AttentionPooling(nn.Module):
    '''
    Graph pooling layer implementing top-k and threshold-based pooling.
    '''
    def __init__(self,
                 in_features,  # feature dimensionality in the current graph layer
                 in_features_prev,  # feature dimensionality in the previous graph layer
                 pool_type,
                 pool_arch,
                 large_graph,
                 attn_gnn=None,
                 kl_weight=None,
                 drop_nodes=True,
                 init='normal',
                 scale=None,
                 debug=False):
        super(AttentionPooling, self).__init__()
        self.pool_type = pool_type
        self.pool_arch = pool_arch
        self.large_graph = large_graph
        self.kl_weight = kl_weight
        self.proj = None
        self.drop_nodes = drop_nodes
        self.is_topk = self.pool_type[2].lower() == 'topk'
        self.scale =scale
        self.init = init
        self.debug = debug
        self.clamp_value = 60
        self.torch = torch.__version__
        if self.is_topk:
            self.topk_ratio = float(self.pool_type[3])  # r
            assert self.topk_ratio > 0 and self.topk_ratio <= 1, ('invalid top-k ratio', self.topk_ratio, self.pool_type)
        else:
            self.threshold = float(self.pool_type[3])  # \tilde{alpha}
            assert self.threshold >= 0 and self.threshold <= 1, ('invalid pooling threshold', self.threshold, self.pool_type)

        if self.pool_type[1] in ['unsup', 'sup']:
            assert self.pool_arch not in [None, 'None'], self.pool_arch

            n_in = in_features_prev if self.pool_arch[1] == 'prev' else in_features
            if self.pool_arch[0] == 'fc':
                p_optimal = torch.from_numpy(np.pad(np.array([0, 1]), (0, n_in - 2), 'constant')).float().view(1, n_in)
                if len(self.pool_arch) == 2:
                    # single layer projection
                    self.proj = nn.Linear(n_in, 1, bias=False)
                    p = self.proj.weight.data
                    if scale is not None:
                        if init == 'normal':
                            p = torch.randn(n_in)  # std=1, seed 9753 for optimal initialization
                        elif init == 'uniform':
                            p = torch.rand(n_in) * 2 - 1  # [-1,1]
                        else:
                            raise NotImplementedError(init)
                        p *= scale  # multiply std for normal or change range for uniform
                    else:
                        print('Default PyTorch init is used for layer %s, std=%.3f' % (str(p.shape), p.std()))
                    self.proj.weight.data = p.view_as(self.proj.weight.data)
                    p = self.proj.weight.data.view(1, n_in)
                else:
                    # multi-layer projection
                    filters = list(map(int, self.pool_arch[2:]))
                    self.proj = []
                    for layer in range(len(filters)):
                        self.proj.append(nn.Linear(in_features=n_in if layer == 0 else filters[layer - 1],
                                                   out_features=filters[layer]))
                        if layer == 0:
                            p = self.proj[0].weight.data
                            if scale is not None:
                                if init == 'normal':
                                    p = torch.randn(filters[layer], n_in)
                                elif init == 'uniform':
                                    p = torch.rand(filters[layer], n_in) * 2 - 1  # [-1,1]
                                else:
                                    raise NotImplementedError(init)
                                p *= scale  # multiply std for normal or change range for uniform
                            else:
                                print('Default PyTorch init is used for layer %s, std=%.3f' % (str(p.shape), p.std()))
                            self.proj[0].weight.data = p.view_as(self.proj[0].weight.data)
                            p = self.proj[0].weight.data.view(-1, n_in)
                            self.proj.append(nn.ReLU(True))

                    self.proj.append(nn.Linear(filters[-1], 1))
                    self.proj = nn.Sequential(*self.proj)

                # Compute cosine similarity with the optimal vector and print values
                # ignore the last dimension, because it does not receive gradients during training
                # n_in=4 for colors-3 because some of our test subsets have 4 dimensional features
                cos_sim = self.cosine_sim(p[:, :-1], p_optimal[:, :-1])
                if p.shape[0] == 1:
                    print('p values', p[0].data.cpu().numpy())
                    print('cos_sim', cos_sim.item())
                else:
                    for fn in [torch.max, torch.min, torch.mean, torch.std]:
                        print('cos_sim', fn(cos_sim).item())
            elif self.pool_arch[0] == 'gnn':
                self.proj = attn_gnn(n_in)
            else:
                raise ValueError('invalid pooling layer architecture', self.pool_arch)

        elif self.pool_type[1] == 'gt':
            if not self.is_topk and self.threshold > 0:
                print('For ground truth attention threshold should be 0, but it is %f' % self.threshold)
        else:
            raise NotImplementedError(self.pool_type[1])

    def __repr__(self):
        return 'AttentionPooling(pool_type={}, pool_arch={}, topk={}, kl_weight={}, init={}, scale={}, proj={})'.format(
            self.pool_type,
            self.pool_arch,
            self.is_topk,
            self.kl_weight,
            self.init,
            self.scale,
            self.proj)

    def cosine_sim(self, a, b):
        return torch.mm(a, b.t()) / (torch.norm(a, dim=1, keepdim=True) * torch.norm(b, dim=1, keepdim=True))

    def mask_out(self, x, mask):
        return x.view_as(mask) * mask

    def drop_nodes_edges(self, x, A, mask):
        N_nodes = torch.sum(mask, dim=1).long()  # B
        N_nodes_max = N_nodes.max()
        idx = None
        if N_nodes_max > 0:
            B, N, C = x.shape
            # Drop nodes
            mask, idx = torch.topk(mask.byte(), N_nodes_max, dim=1, largest=True, sorted=False)
            x = torch.gather(x, dim=1, index=idx.unsqueeze(2).expand(-1, -1, C))
            # Drop edges
            A = torch.gather(A, dim=1, index=idx.unsqueeze(2).expand(-1, -1, N))
            A = torch.gather(A, dim=2, index=idx.unsqueeze(1).expand(-1, N_nodes_max, -1))

        return x, A, mask, N_nodes, idx

    def forward(self, data):

        KL_loss = None
        x, A, mask, _, params_dict = data[:5]

        mask_float = mask.float()
        N_nodes_float = params_dict['N_nodes'].float()
        B, N, C = x.shape
        A = A.view(B, N, N)
        alpha_gt = None
        if 'node_attn' in params_dict:
            if not isinstance(params_dict['node_attn'], list):
                params_dict['node_attn'] = [params_dict['node_attn']]
            alpha_gt = params_dict['node_attn'][-1].view(B, N)
        if 'node_attn_eval' in params_dict:
            if not isinstance(params_dict['node_attn_eval'], list):
                params_dict['node_attn_eval'] = [params_dict['node_attn_eval']]

        if (self.pool_type[1] == 'gt' or (self.pool_type[1] == 'sup' and self.training)) and alpha_gt is None:
            raise ValueError('ground truth node attention values node_attn required for %s' % self.pool_type)

        if self.pool_type[1] in ['unsup', 'sup']:
            attn_input = data[-1] if self.pool_arch[1] == 'prev' else x.clone()
            if self.pool_arch[0] == 'fc':
                alpha_pre = self.proj(attn_input).view(B, N)
            else:
                # to support python2
                input = [attn_input]
                input.extend(data[1:])
                alpha_pre = self.proj(input)[0].view(B, N)
            # softmax with masking out dummy nodes
            alpha_pre = torch.clamp(alpha_pre, -self.clamp_value, self.clamp_value)
            alpha = normalize_batch(self.mask_out(torch.exp(alpha_pre), mask_float).view(B, N))
            if self.pool_type[1] == 'sup' and self.training:
                if self.torch.find('1.') == 0:
                    KL_loss_per_node = self.mask_out(F.kl_div(torch.log(alpha + 1e-14), alpha_gt, reduction='none'),
                                                     mask_float.view(B,N))
                else:
                    KL_loss_per_node = self.mask_out(F.kl_div(torch.log(alpha + 1e-14), alpha_gt, reduce=False),
                                                     mask_float.view(B, N))
                KL_loss = self.kl_weight * torch.mean(KL_loss_per_node.sum(dim=1) / (N_nodes_float + 1e-7))  # mean over nodes, then mean over batches
        else:
            alpha = alpha_gt

        x = x * alpha.view(B, N, 1)
        if self.large_graph:
            # For large graphs during training, all alpha values can be very small hindering training
            x = x * N_nodes_float.view(B, 1, 1)
        if self.is_topk:
            N_remove = torch.round(N_nodes_float * (1 - self.topk_ratio)).long()  # number of nodes to be removed for each graph
            idx = torch.sort(alpha, dim=1, descending=False)[1]  # indices of alpha in ascending order
            mask = mask.clone().view(B, N)
            for b in range(B):
                idx_b = idx[b, mask[b, idx[b]]]  # take indices of non-dummy nodes for current data example
                mask[b, idx_b[:N_remove[b]]] = 0
        else:
            mask = (mask & (alpha.view_as(mask) > self.threshold)).view(B, N)

        if self.drop_nodes:
            x, A, mask, N_nodes_pooled, idx = self.drop_nodes_edges(x, A, mask)
            if idx is not None and 'node_attn' in params_dict:
                # update ground truth (or weakly labeled) attention for a reduced graph
                params_dict['node_attn'].append(normalize_batch(self.mask_out(torch.gather(alpha_gt, dim=1, index=idx), mask.float())))
            if idx is not None and 'node_attn_eval' in params_dict:
                # update ground truth (or weakly labeled) attention for a reduced graph
                params_dict['node_attn_eval'].append(normalize_batch(self.mask_out(torch.gather(params_dict['node_attn_eval'][-1], dim=1, index=idx), mask.float())))
        else:
            N_nodes_pooled = torch.sum(mask, dim=1).long()  # B
            if 'node_attn' in params_dict:
                params_dict['node_attn'].append((self.mask_out(params_dict['node_attn'][-1], mask.float())))
            if 'node_attn_eval' in params_dict:
                params_dict['node_attn_eval'].append((self.mask_out(params_dict['node_attn_eval'][-1], mask.float())))

        params_dict['N_nodes'] = N_nodes_pooled

        mask_matrix = mask.unsqueeze(2) & mask.unsqueeze(1)
        A = A * mask_matrix.float()   # or A[~mask_matrix] = 0

        # Add additional losses regularizing the model
        if KL_loss is not None:
            if 'reg' not in params_dict:
                params_dict['reg'] = []
            params_dict['reg'].append(KL_loss)

        # Keep attention coefficients for evaluation
        for key, value in zip(['alpha', 'mask'], [alpha, mask]):
            if key not in params_dict:
                params_dict[key] = []
            params_dict[key].append(value.detach())

        if self.debug and alpha_gt is not None:
            idx_correct_pool = (alpha_gt > 0)
            idx_correct_drop = (alpha_gt == 0)
            alpha_correct_pool = alpha[idx_correct_pool].sum() / N_nodes_float.sum()
            alpha_correct_drop = alpha[idx_correct_drop].sum() / N_nodes_float.sum()
            ratio_avg = (N_nodes_pooled.float() / N_nodes_float).mean()

            for key, values in zip(['alpha_correct_pool_debug', 'alpha_correct_drop_debug', 'ratio_avg_debug'],
                                  [alpha_correct_pool, alpha_correct_drop, ratio_avg]):
                if key not in params_dict:
                    params_dict[key] = []
                params_dict[key].append(values.detach())

        output = [x, A, mask]
        output.extend(data[3:])
        return output
