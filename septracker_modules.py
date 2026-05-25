import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class AddAndNorm1(nn.Module):
    def __init__(self, dim, drop=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, residual):
        return self.norm(self.drop(x) + residual)

def col_softmax(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    
    return F.softmax(x, dim=dim)

def entropy(p: torch.Tensor, dim: int, eps: float = 1e-9) -> torch.Tensor:
    
    p = p.clamp(min=eps)
    return -(p * p.log()).sum(dim=dim)

def symmetrize(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(0, 1))

def _pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
    x_n = F.normalize(x, dim=-1)
    return x_n @ x_n.t()

def _cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_n = F.normalize(x, dim=-1)
    y_n = F.normalize(y, dim=-1)
    return x_n @ y_n.t()

class HGATDualSpaceLite(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        heads = int(getattr(cfg.GAT, "DS_HEADS", 1))
        d_in, d_hid, d_out = list(getattr(cfg.GAT, "DS_DIM", [128, 128, 128]))
        drop = float(getattr(cfg.GAT, "DROP", 0.6))
        self.k_nb = int(getattr(cfg.GAT, "DS_K_NB",8))
        self.tau_node = float(getattr(cfg.GAT, "TDS_AU_NODE", 0.7))
        self.tau_proto = float(getattr(cfg.GAT, "DS_TAU_PROTO", 0.7))
        self.proto_num = int(getattr(cfg.GAT, "DS_PROTO_M", 16))
        
        self.heads = heads
        assert d_hid % heads == 0, "d_hid must be divisible by heads"
        self.d_in, self.d_hid, self.d_out = d_in, d_hid, d_out
        self.d_head = d_hid // heads

        self.softmax_tau = 1.0

        self.in_linear  = nn.Linear(d_in,  d_hid)
        self.mid_linear = nn.Linear(d_hid, d_out)
        self.out_proj   = nn.Linear(d_hid, d_out)
        self.add_norm1  = AddAndNorm1(d_hid, drop)
        self.add_norm2  = AddAndNorm1(d_out, drop)

        self.node_Wn = nn.ModuleList([
            nn.Linear(self.d_head, self.d_head, bias=False) for _ in range(heads)
        ])
        self.node_We = nn.ModuleList([
            nn.Linear(self.d_head, self.d_head, bias=False) for _ in range(heads)
        ])

        self.proto_Wn = nn.ModuleList([
            nn.Linear(self.d_head, self.d_head, bias=False) for _ in range(heads)
        ])
        self.proto_We = nn.ModuleList([
            nn.Linear(self.d_head, self.d_head, bias=False) for _ in range(heads)
        ])
        self.prototypes = nn.ParameterList([
            nn.Parameter(torch.randn(self.proto_num, self.d_head) * (1.0 / math.sqrt(self.d_head)))     
            for _ in range(heads)
        ])

        self.fusion_scorer = nn.ModuleList([
            nn.Linear(self.d_head, 2, bias=True) for _ in range(heads)
        ])

    @torch.no_grad()
    def _S_node(self, x_head: torch.Tensor) -> torch.Tensor:
        """节点中心分支"""
        N, _ = x_head.shape
        sim = _pairwise_cosine(x_head)
        sim = sim - torch.eye(N, device=x_head.device,
                              dtype=x_head.dtype) * 1e9
        k = min(self.k_nb, N)
        topk = torch.topk(sim, k=k, dim=0)
        idx, val = topk.indices, topk.values
        S = torch.full((N, N), -1e9, device=x_head.device,
                       dtype=x_head.dtype)
        for h in range(N):
            rows = idx[:k, h]
            S[rows, h] = val[:k, h] / self.tau_node
        return F.softmax(S, dim=0)

    def _S_proto_col(self, x_head: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """原型分支"""
        sim = _cosine(x_head, P)
        S_row = F.softmax(sim / self.tau_proto, dim=1)
        col_sum = S_row.sum(dim=0, keepdim=True) + 1e-9
        return S_row / col_sum

    def _head_forward(self, x_head: torch.Tensor, hi: int) -> torch.Tensor:
       
        x_node = self.node_Wn[hi](x_head)
        S_node = self._S_node(x_node)
        E_node = S_node.t() @ x_node
        E_node = self.node_We[hi](E_node)
        Y_node = S_node @ E_node

        x_proto = self.proto_Wn[hi](x_head)
        P = self.prototypes[hi]
        S_proto = self._S_proto_col(x_proto, P)
        E_proto = S_proto.t() @ x_proto
        E_proto = self.proto_We[hi](E_proto)
        Y_proto = S_proto @ E_proto

        logits = self.fusion_scorer[hi](x_head) / max(self.softmax_tau, 1e-6)
        w = F.softmax(logits, dim=-1)
        w_node  = w[:, :1]
        w_proto = w[:, 1:]
        return w_node * Y_node + w_proto * Y_proto

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        
        x = x.to(torch.float32)
        x_in = self.in_linear(x)
        heads_in = torch.chunk(x_in, self.heads, dim=-1)

        heads_out = [self._head_forward(hx, i)
                     for i, hx in enumerate(heads_in)]
        y_cat = torch.cat(heads_out, dim=-1)

        y = self.add_norm1(y_cat, x_in)
        y = F.relu(y)
        y_out = self.out_proj(y)
        y_out = self.add_norm2(y_out, self.mid_linear(y))
        return y_out

class LookupTableLayer(nn.Module):
    
    def __init__(self, length, dimension, cfg):
        super(LookupTableLayer, self).__init__()
        self.cfg = cfg
        self.rand_num = cfg.PARAM.EMPTY_NUM
        self.empty_index = cfg.PARAM.MAX_LEN + 1
        self.table_len = length + self.rand_num
        self.dimension = dimension
        self.initiate_mode = cfg.EXTRACT.INIT

        table_tensor = self.initiate_by_position(length, cfg).to(cfg.LOAD.DEV)
        self.fixed_lookup_table = table_tensor.to(torch.float32)

        if self.initiate_mode == 'random':
            self.lookup_table_x = torch.nn.Parameter(torch.randn(self.table_len, dimension)).to(torch.float32)
            self.lookup_table_y = torch.nn.Parameter(torch.randn(self.table_len, dimension)).to(torch.float32)
        elif self.initiate_mode == 'position':
            self.lookup_table_x = torch.nn.Parameter(table_tensor).to(torch.float32)
            self.lookup_table_y = torch.nn.Parameter(table_tensor).to(torch.float32)

    def forward(self, positions):
        if positions.ndim != 3 and positions.ndim != 4:
            raise ValueError('positions dim should be 3 for detection or 4 for tracks, got {}'.format(positions.ndim))

        elif positions.ndim == 4:
            batch_size, max_len, rewind_len, position_len = positions.shape
            positions_slices = positions.chunk(2, dim=3)
            lookup_tables = [self.lookup_table_x, self.lookup_table_y]
            total_len = batch_size * max_len * rewind_len
            result_tensors = []
            max_value_x = torch.max(self.lookup_table_x)
            max_value_y = torch.max(self.lookup_table_y)
            max_values = [max_value_x, max_value_y]
            for positions_slice, org_lookup_table_slice, max_value in zip(positions_slices, lookup_tables, max_values):
                std_lookup_table_slice = org_lookup_table_slice / max_value
                lookup_table_slice = self.cfg.EXTRACT.ADD_RATE * std_lookup_table_slice + self.fixed_lookup_table
                flattened_positions = positions_slice.reshape(total_len)
                flattened_positions = torch.where(flattened_positions < 0, torch.tensor(1., dtype=flattened_positions.dtype).to(self.cfg.LOAD.DEV),
                                                  flattened_positions)
                concat_positions = torch.arange(flattened_positions.shape[0]).to(self.cfg.LOAD.DEV)
                coalesced_indices = torch.stack([concat_positions, flattened_positions])
                values = torch.ones_like(concat_positions)
                sparse_positions = torch.sparse_coo_tensor(coalesced_indices, values, size=(total_len, self.table_len)).to(torch.float32)
                encode_result = torch.sparse.mm(sparse_positions, lookup_table_slice)
                position_embeddings = encode_result.reshape((batch_size, max_len, rewind_len, self.dimension))
                result_tensors.append(position_embeddings)

            concat_position_embeddings = torch.cat(tuple(result_tensors), dim=-1)
            shortened_concat_position_embeddings = concat_position_embeddings.view(batch_size, max_len, rewind_len, self.dimension, 2).sum(dim=-1)

        elif positions.ndim == 3:
            batch_size, max_len, position_len = positions.shape
            positions_slices = positions.chunk(2, dim=2)
            lookup_tables = [self.lookup_table_x, self.lookup_table_y]
            max_value_x = torch.max(self.lookup_table_x)
            max_value_y = torch.max(self.lookup_table_y)
            max_values = [max_value_x, max_value_y]
            total_len = batch_size * max_len
            result_tensors = []
            for positions_slice, org_lookup_table_slice, max_value in zip(positions_slices, lookup_tables, max_values):
                std_lookup_table_slice = org_lookup_table_slice / max_value
                lookup_table_slice = self.cfg.EXTRACT.ADD_RATE * std_lookup_table_slice + self.fixed_lookup_table
                flattened_positions = positions_slice.reshape(total_len)
                flattened_positions = torch.where(flattened_positions < 0, torch.tensor(1., dtype=flattened_positions.dtype).to(self.cfg.LOAD.DEV),
                                                  flattened_positions)
                concat_positions = torch.arange(flattened_positions.shape[0]).to(self.cfg.LOAD.DEV)
                coalesced_indices = torch.stack([concat_positions, flattened_positions])
                values = torch.ones_like(concat_positions)
                sparse_positions = torch.sparse_coo_tensor(coalesced_indices, values, size=(total_len, self.table_len)).to(
                    torch.float32)
                encode_result = torch.sparse.mm(sparse_positions, lookup_table_slice)
                position_embeddings = encode_result.reshape((batch_size, max_len, self.dimension))
                result_tensors.append(position_embeddings)

            concat_position_embeddings = torch.cat(tuple(result_tensors), dim=-1)
            shortened_concat_position_embeddings = concat_position_embeddings.view(batch_size, max_len,
                                                                                   self.dimension, 2).sum(dim=-1)

        return shortened_concat_position_embeddings

    def old_initiate_by_position(self, length, cfg):
        tensor_1 = torch.arange(length, dtype=torch.float) / torch.tensor(1000) - torch.tensor(0.5)
        tensor_2 = torch.full((self.rand_num,), -5, dtype=torch.float)
        result_tensor = torch.cat((tensor_1, tensor_2), dim=0)
        final_tensor = result_tensor.unsqueeze(1).expand(-1, self.dimension)

        return final_tensor

    def initiate_by_position(self, length, cfg):
        start = cfg.EXTRACT.START
        pos_step = cfg.EXTRACT.POS_STEP
        embed_step = cfg.EXTRACT.EMB_STEP
        empty_value = cfg.EXTRACT.SUP_VAL
        tensor_list = []
        for i in range(self.dimension):
            tensor_start = start + i * embed_step
            tensor_values = torch.arange(tensor_start, tensor_start + length * pos_step, pos_step)
            tensor_list.append(tensor_values.unsqueeze(1))
        tensor_1 = torch.cat(tensor_list, dim=1)
        tensor_2 = torch.full((self.rand_num, self.dimension), empty_value)
        result = torch.cat((tensor_1, tensor_2), dim=0)

        return result

class ExtractionModel(nn.Module):
    
    def __init__(self, length, cfg):
        dimension = cfg.EXTRACT.DIM
        super(ExtractionModel, self).__init__()
        self.lookup_table = LookupTableLayer(length, dimension, cfg)
        self.linear = nn.Linear(2*dimension, dimension)
        self.linear_for_val = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(cfg.EXTRACT.DROP)

    def forward(self, positions_all):

        position_embeddings = self.lookup_table(positions_all)
        
        return position_embeddings

class AddAndNorm(nn.Module):
    
    def __init__(self, hidden_dim, dropout_rate):
        super(AddAndNorm, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm_1d = nn.BatchNorm1d(hidden_dim)
        self.batch_norm_2d = nn.BatchNorm2d(hidden_dim)

    def forward(self, x, residual):
        x = self.dropout(x)
        x = x + residual
        x = self.batch_norm_1d(x)
        return x

class GATLayer(torch.nn.Module):
    
    def __init__(self, cfg):
        super(GATLayer, self).__init__()
        heads = cfg.GAT.HEADS
        in_features, hidden_dim, out_features = cfg.GAT.DIM
        self.conv1 = GATConv(in_features, int(hidden_dim / heads), heads=heads)
        self.conv2 = GATConv(hidden_dim, out_features, heads=1)
        self.add_norm = AddAndNorm(hidden_dim, cfg.GAT.DROP)

    def forward(self, x, edge_index):
        x_residual = x.clone()
        x = self.conv1(x, edge_index)
        x = self.add_norm(x, x_residual)
        x = F.relu(x)
        x_residual = x.clone()
        x = self.conv2(x, edge_index)
        x = self.add_norm(x, x_residual)
        return x

class TransformerEncoder(nn.Module):
    
    def __init__(self, cfg):
        super(TransformerEncoder, self).__init__()
        input_dim = cfg.TF.INPUT_DIM
        dim_ffn = cfg.TF.FFN_DIM
        num_layers = cfg.TF.ENCODER_NUM
        num_heads = cfg.TF.HEADS
        dropout_rate = cfg.TF.DROP
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_ffn,
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

    def forward(self, embeddings):
        embeddings = embeddings.permute(1, 0, 2)
        encoder_output = self.transformer_encoder(embeddings)
        return encoder_output.permute(1, 0, 2)

class TransformerDecoder(nn.Module):
    
    def __init__(self, cfg):
        super(TransformerDecoder, self).__init__()
        input_dim = cfg.TF.INPUT_DIM
        dim_ffn = cfg.TF.FFN_DIM
        num_layers = cfg.TF.DECODER_NUM
        num_heads = cfg.TF.HEADS
        dropout_rate = cfg.TF.DROP
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_ffn,
            dropout=dropout_rate
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)

    def forward(self, embeddings, encoder_output):
        embeddings = embeddings.permute(1, 0, 2)
        encoder_output = encoder_output.permute(1, 0, 2)
        decoder_output = self.transformer_decoder(embeddings, encoder_output)
        return decoder_output.permute(1, 0, 2)

class ClassifyLayer(torch.nn.Module):
    def __init__(self, cfg):
        super(ClassifyLayer, self).__init__()
        self.device = cfg.LOAD.DEV

    def forward(self, embeddings, W, len_prd, head=1):
        
        mask = W[:len_prd, len_prd:] if head == 1 else W
        mask = mask.float()
        mask_org = torch.ones_like(mask).to(self.device)
        mask = -99 * (mask_org - mask)

        norms = torch.norm(embeddings, dim=1, keepdim=True)
        for i in range(len(norms)):
            if norms[i] <= 1e-9:
                norms[i] = 1e-9
        normalized_matrix = embeddings / norms
        similarity_matrix = torch.matmul(normalized_matrix, normalized_matrix.t())
        similarity_matrix = similarity_matrix[:len_prd, len_prd:] if head == 1 else similarity_matrix
        masked_similarity_matrix = similarity_matrix + mask
        x_len, y_len = masked_similarity_matrix.shape
        inner_matrix = torch.ones((x_len, y_len)).to(self.device)
        score_matrix = inner_matrix - masked_similarity_matrix
        output_matrix = torch.relu(masked_similarity_matrix)
        output_matrix = torch.clamp(output_matrix, max=1)
        half_normalized_matrix = torch.div(torch.ones((x_len, y_len)).to(self.device), score_matrix)
        row_sum = torch.sum(half_normalized_matrix, dim=1, keepdim=True)

        for i in range(len(row_sum)):
            if row_sum[i] <= 1e-9:
                row_sum[i] = 1e-9
        normalized_matrix = half_normalized_matrix / row_sum

        supple_tensor = torch.tensor([0 if torch.any(row > 0) else 1 for row in masked_similarity_matrix]).to(self.device)
        processed_matrix = normalized_matrix.clone()
        for i, val in enumerate(supple_tensor):
            if val.item() == 1:
                processed_matrix[i] = 0
        supple_tensor_transposed = supple_tensor.view(-1, 1)
        extended_matrix = torch.cat((processed_matrix, supple_tensor_transposed), dim=1)

        return output_matrix, extended_matrix

