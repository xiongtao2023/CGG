import torch as th
from torch import nn
import math
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
import copy

from cggframe import CGGframe
from cagg import CGGLayer
from utils.data.collate import CollateFnGNN1
from utils.Dict import Dict
from utils.prepare_batch import prepare_batch_factory_recursive
from utils.data.transform import seq_to_weighted_graph


class PointWiseFeedForward(th.nn.Module):
    def __init__(self, embedding_dim, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = th.nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        self.dropout1 = th.nn.Dropout(p=dropout_rate)
        self.relu = th.nn.ReLU()
        self.conv2 = th.nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        self.dropout2 = th.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class Attention(nn.Module):
    def __init__(self, hidden_dim, session_len):
        super(Attention, self).__init__()
        self.attn_w0 = nn.Parameter(th.Tensor(session_len, hidden_dim))
        self.attn_w1 = nn.Parameter(th.Tensor(hidden_dim, hidden_dim))
        self.attn_w2 = nn.Parameter(th.Tensor(hidden_dim, hidden_dim))
        self.attn_bias = nn.Parameter(th.Tensor(hidden_dim))
        self.initial_()

    def initial_(self):
        init.normal_(self.attn_w0, 0, 0.05)
        init.normal_(self.attn_w1, 0, 0.05)
        init.normal_(self.attn_w2, 0, 0.05)
        init.constant_(self.attn_bias, 0)

    def forward(self, q, k, v, mask=None, dropout=None):
        alpha = th.matmul(
            th.relu(k.matmul(self.attn_w1) + q.matmul(self.attn_w2) + self.attn_bias),
            self.attn_w0.t(),
        )  # (B,seq,1)

        if mask is not None:
            alpha = alpha.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(alpha, dim=-2)
        if dropout is not None:
            alpha = dropout(alpha)
        re = th.matmul(alpha.transpose(-1, -2), v)  # (B, 1, dim)
        return re


class MultiHeadedAttention(nn.Module):
    def __init__(self, hidden_dim, session_len, num_heads=1, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert hidden_dim % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = hidden_dim // num_heads
        self.num_heads = num_heads

        self.attn = Attention(self.d_k, session_len)
        self.linears = self.clones(nn.Linear(hidden_dim, hidden_dim), 4)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def clones(self, module, N):
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def forward(self, query, key, value, mask=None, linear=False):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        if linear is True:
            query, key, value = [lin(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2) for lin, x in zip(self.linears, (query, key, value))]
        else:
            query, key, value = [x.view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2) for x in (query, key, value)]
        # (batch,num_heads,max_len,hidden_dim/num_heads)
        # 2) Apply attention on all the projected vectors in batch.
        x = self.attn(query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        return self.linears[-1](x) if linear is True else x


class CGG(CGGframe):
    def __init__(
        self,
        num_categorys,
        num_items,
        max_len,
        num_heads,
        num_blocks,
        dropout_rate,
        embedding_dim,
        knowledge_graph,
        num_layers,
        relu=False,
        batch_norm=True,
        feat_drop=0.0,
        beta1=0.01,
        beta2=0.01,
        **kwargs
    ):
        super().__init__(
            num_categorys,
            num_items,
            embedding_dim,
            knowledge_graph,
            num_layers,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
        )
        self.pad_embedding = nn.Embedding(1, embedding_dim, max_norm=1)
        self.pad_indices = nn.Parameter(th.arange(1, dtype=th.long), requires_grad=False)
        self.pos_embedding = nn.Embedding(max_len, embedding_dim, max_norm=1)

        self.att_i = nn.Linear(embedding_dim * 4, embedding_dim * 2, bias=False)
        self.att_c = nn.Linear(embedding_dim * 4, embedding_dim * 2, bias=False)

        self.fc_i = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_c = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_sr_i = nn.Linear(embedding_dim*4, embedding_dim*2, bias=False)
        self.fc_sr_c = nn.Linear(embedding_dim * 4, embedding_dim * 2, bias=False)
        self.PSE_layer = CGGLayer(
            embedding_dim,
            num_steps=1,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            relu=relu,
        )
        input_dim = 4 * embedding_dim
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.fc_sr = nn.Linear(input_dim, embedding_dim, bias=False)

        self.mul = MultiHeadedAttention(2 * embedding_dim, max_len, num_heads)
        self.mul1 = MultiHeadedAttention(2 * embedding_dim, 1, num_heads)
        self.dropout = nn.Dropout(0.5)
        # SAS
        self.attention_layernorms = th.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = th.nn.ModuleList()
        self.forward_layernorms = th.nn.ModuleList()
        self.forward_layers = th.nn.ModuleList()

        self.last_layernorm = th.nn.LayerNorm(2*embedding_dim, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = th.nn.LayerNorm(2*embedding_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = th.nn.MultiheadAttention(2*embedding_dim, num_heads, dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = th.nn.LayerNorm(2*embedding_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(2*embedding_dim, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        self.feed = PointWiseFeedForward(embedding_dim*2, 0.5)

        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.w1 = nn.Parameter(th.Tensor(embedding_dim*2, 1))
        self.w2 = nn.Parameter(th.Tensor(embedding_dim * 2, 1))
        self.beta1 = beta1
        self.beta2 = beta2

    def SSL(self, sess_emb_hgnn, sess_emb_lgcn):
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[th.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,th.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return th.sum(th.mul(x1, x2), 1)
        pos = score(sess_emb_hgnn, sess_emb_lgcn)
        neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))
        one = th.FloatTensor(neg1.shape[0]).fill_(1)
        # one = zeros = torch.ones(neg1.shape[0])
        con_loss = th.sum(-th.log(1e-8 + th.sigmoid(pos))-th.log(1e-8 + (one - th.sigmoid(neg1))))
        return con_loss

    def forward(self, inputs, extra_inputs=None):
        KG_embeddings = super().forward(extra_inputs)

        KG_embeddings["item"] = th.cat([KG_embeddings["item"], self.pad_embedding(self.pad_indices)], dim=0)
        KG_embeddings["category"] = th.cat([KG_embeddings["category"], self.pad_embedding(self.pad_indices)], dim=0)

        # Hierarchical sequential session interest encoders
        padded_seqs, padded_seqs_ca, pos, g, g_ca = inputs

        emb_seqs = KG_embeddings["item"][padded_seqs]
        emb_seqs_ca = KG_embeddings["category"][padded_seqs_ca]
        pos_emb = self.pos_embedding(pos)

        feat_1 = th.cat([emb_seqs, pos_emb.unsqueeze(0).expand(emb_seqs.shape)], dim=-1)
        tl = feat_1.shape[1]  # time dim len for enforce causality
        attention_mask = ~th.tril(th.ones((tl, tl), dtype=th.bool))

        for i in range(len(self.attention_layers)):
            feat_1 = th.transpose(feat_1, 0, 1)
            Q = self.attention_layernorms[i](feat_1)
            mha_outputs, _ = self.attention_layers[i](Q, feat_1, feat_1,
                                                      attn_mask=attention_mask)
            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?
            feat_1 = Q + mha_outputs
            feat_1 = th.transpose(feat_1, 0, 1)

            feat_1 = self.forward_layernorms[i](feat_1)
            feat_1 = self.forward_layers[i](feat_1)
            # feat_1 *= ~timeline_mask.unsqueeze(-1)

        log_feats_1 = self.last_layernorm(feat_1)

        feat_2 = th.cat([emb_seqs_ca, pos_emb.unsqueeze(0).expand(emb_seqs_ca.shape)], dim=-1)
        tl = feat_2.shape[1]
        attention_mask = ~th.tril(th.ones((tl, tl), dtype=th.bool))

        for i in range(len(self.attention_layers)):
            feat_2 = th.transpose(feat_2, 0, 1)
            Q = self.attention_layernorms[i](feat_2)
            mha_outputs, _ = self.attention_layers[i](Q, feat_2, feat_2,
                                                      attn_mask=attention_mask)
            feat_2 = Q + mha_outputs
            feat_2 = th.transpose(feat_2, 0, 1)

            feat_2 = self.forward_layernorms[i](feat_2)
            feat_2 = self.forward_layers[i](feat_2)
            # feat_2 *= ~timeline_mask.unsqueeze(-1)

        log_feats_2 = self.last_layernorm(feat_2)  # (U, T, C) -> (U, -1, C)

        attn_item = th.squeeze(self.pool(log_feats_1), dim=1)
        attn_cate = th.squeeze(self.pool(log_feats_2), dim=1)
        # alpha1 = th.matmul(log_feats_1, self.w1)
        # attn_item = th.sum(alpha1 * log_feats_1, 1)
        # alpha2 = th.matmul(log_feats_2, self.w2)
        # attn_cate = th.sum(alpha2 * log_feats_2, 1)

        # Hierarchical graphical session interest encoders
        iids = g.ndata['iid']  # (num_nodes,)
        cids = g_ca.ndata['iid']
        feat_i = KG_embeddings['item'][iids]
        feat_c = KG_embeddings['category'][cids]

        feat_item = self.fc_i(feat_i)
        feat_category = self.fc_c(feat_c)
        feat_i = self.PSE_layer(g, feat_item)
        feat_c = self.PSE_layer(g_ca, feat_category)

        # Dual-pattern contrastive learning
        con_loss_i = self.SSL(feat_i, attn_item)
        con_loss_c = self.SSL(feat_c, attn_cate)
        # sr_item = th.add(attn_item, feat_i)
        sr_item = th.cat([attn_item, feat_i], dim=1)
        sr_item = self.fc_sr_i(sr_item)
        # sr_category = th.add(attn_cate, feat_c)
        sr_category = th.cat([attn_cate, feat_c], dim=1)
        sr_category = self.fc_sr_c(sr_category)
        sr = th.cat([sr_item, sr_category], dim=1)
        sr = self.fc_sr(sr)
        # if self.batch_norm is not None:
        #     sr = self.batch_norm(sr)
        logits = sr @ self.item_embedding(self.item_indices).t()

        return sr, logits, self.beta1*con_loss_i + self.beta2*con_loss_c


seq_to_graph_fns = [seq_to_weighted_graph]

config = Dict({
    'Model': CGG,
    'seq_to_graph_fns': seq_to_graph_fns,
    'CollateFn': CollateFnGNN1,
    'prepare_batch_factory': prepare_batch_factory_recursive,
})
