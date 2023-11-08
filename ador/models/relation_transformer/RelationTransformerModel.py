##########################################################
# Copyright 2019 Oath Inc.
# Licensed under the terms of the MIT license.
# Please see LICENSE file in the project root for terms.
##########################################################
# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
# import misc.utils as utils
# utils.PositionalEmbedding()
import copy
import math
import numpy as np

from .CaptionModel import CaptionModel
# from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from utils_.drop_path import DropPath


def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding=True):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image

    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j

    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    # returns a relational embedding for each pair of bboxes, with dimension = dim_g
    # follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    batch_size = f_g.size(0)

    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=-1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)


def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    sorted_lengths = sorted_lengths.cpu()
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, frame_embed=None, segment_embed=None):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        # self.generator = generator
        self.frame_embed = frame_embed
        self.segment_embed = segment_embed

    def forward(self, src, boxes, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, boxes, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, boxes, src_mask, memory=None, not_objects=None):
        return self.encoder(self.src_embed(src), boxes, src_mask,
                            memory=memory, not_objects=not_objects)

    def decode(self, memory, src_mask, tgt, tgt_mask, frame_features=None, segment_labels=None):
        embeddings = self.tgt_embed(tgt)
        if frame_features is not None:
            assert self.frame_embed is not None
            num_clips, num_mf = frame_features.shape[0:2]
            frame_features = frame_features.view(-1, 1024, 7, 7)
            frame_features = self.frame_embed(frame_features)
            frame_features = frame_features.view(num_clips, num_mf, 512)
            # frame_features = self.frame_embed(frame_features)
            frame_features += self.segment_embed(segment_labels)
            embeddings = torch.cat([embeddings, frame_features], dim=1)
        return self.decoder(embeddings, memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab, normalize_out):
        super(Generator, self).__init__()
        self.normalize_out = normalize_out
        self.proj = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x):
        # return F.log_softmax(self.proj(x), dim=-1)
        # return F.sigmoid(self.proj(x))

        if self.normalize_out:
            for W in self.proj.parameters():
                W = F.normalize(W, dim=1)

            x = F.normalize(x, dim=1)

        return self.proj(x)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNorm(layers[0].size)

    def forward(self, x, box, mask, memory=None, not_objects=None):
        "Pass the input (and mask) through each layer in turn."
        for i, layer in enumerate(self.layers):
            # if memory is not None:
            #     x = layer(x, box, mask, frame_features=frame_features[i], not_objects=not_objects)
            # else:
            x = layer(x, box, mask, memory=memory, not_objects=not_objects)

        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout, drop_path):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.drop_path(self.dropout(sublayer(self.norm(x))))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, attn, attn_type, feed_forward, dropout, drop_path):
        super(EncoderLayer, self).__init__()
        if attn_type == 'box':
            self.box_attn = attn  # in order to load pretrained weights
        elif attn_type == 'self':
            self.attn = attn

        self.attn_type = attn_type
        self.feed_forward = feed_forward

        self.sublayer = clones(SublayerConnection(size, dropout, drop_path), 2)

        self.size = size

    def forward(self, x, box, mask, memory=None, not_objects=None):
        "Follow Figure 1 (left) for connections."
        # x = self.sublayer[0](x, lambda x: self.box_attn(x, x, x, box, mask, not_objects=not_objects))
        m = memory if memory is not None else x
        if self.attn_type == 'self':
            x = self.sublayer[0](x, lambda x: self.attn(x, m, m, mask))
        elif self.attn_type == 'box':
            x = self.sublayer[0](x, lambda x: self.box_attn(x, m, m, box, mask, not_objects=not_objects))
        else:
            raise Exception
        # if frame_features is not None:
        #     x = self.sublayer[2](x, lambda x: self.self_attn(x, frame_features, frame_features, mask.transpose(1, 2)))
        #     # x = self.sublayer[1](x, lambda x: self.self_attn(frame_features, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, drop_path):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, drop_path), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        if self.self_attn is not None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # p_attn = scores.transpose(2, 3)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
    # return torch.mul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def box_attention(query, key, value, box_relation_embds_matrix, mask=None, dropout=None, not_objects=None):
    '''
    Compute 'Scaled Dot Product Attention as in paper Relation Networks for Object Detection'.
    Follow the implementation in https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1026-L1055
    '''

    N = value.size()[:2]
    dim_k = key.size(-1)
    dim_g = box_relation_embds_matrix.size()[-1]

    w_q = query
    w_k = key.transpose(-2, -1)
    w_v = value

    # attention weights
    scaled_dot = torch.matmul(w_q, w_k)
    scaled_dot = scaled_dot / np.sqrt(dim_k)
    if mask is not None:
        scaled_dot = scaled_dot.masked_fill(mask == 0, -1e9)

    # w_g = box_relation_embds_matrix.view(N,N)
    w_g = box_relation_embds_matrix
    w_a = scaled_dot
    # w_a = scaled_dot.view(N,N)
    mask_obj = torch.ones_like(w_g).cuda()
    if not_objects is not None:
        assert not_objects.dtype == torch.bool
        num_heads, num_elements = value.shape[1:3]
        # m_obj = not_objects.view(-1, num_elements).unsqueeze(1).unsqueeze(3).repeat(1, num_heads, 1, num_elements)
        mask_obj = not_objects[:, None, :, None].repeat(1, num_heads, 1, num_elements)
        mask_obj += mask_obj.transpose(2, 3).clone()
        mask_obj = 1.0 - mask_obj.float()

    # multiplying log of geometric weights by feature weights
    w_mn = torch.log(torch.clamp(w_g, min=1e-6))*mask_obj + w_a
    w_mn = torch.nn.Softmax(dim=-1)(w_mn)
    if dropout is not None:
        w_mn = dropout(w_mn)

    output = torch.matmul(w_mn, w_v)

    return output, w_mn


class BoxMultiHeadedAttention(nn.Module):
    '''
    Self-attention layer with relative position weights.
    Following the paper "Relation Networks for Object Detection" in https://arxiv.org/pdf/1711.11575.pdf
    '''

    def __init__(self, h, d_model, trignometric_embedding=True, legacy_extra_skip=False, dropout=0.1):
        "Take in model size and number of heads."
        super(BoxMultiHeadedAttention, self).__init__()

        assert d_model % h == 0
        self.trignometric_embedding = trignometric_embedding
        self.legacy_extra_skip = legacy_extra_skip

        # We assume d_v always equals d_k
        self.h = h
        self.d_k = d_model // h
        if self.trignometric_embedding:
            self.dim_g = 64
        else:
            self.dim_g = 4
        geo_feature_dim = self.dim_g

        # matrices W_q, W_k, W_v, and one last projection layer
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.WGs = clones(nn.Linear(geo_feature_dim, 1, bias=True), 8)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_query, input_key, input_value, input_box, mask=None, not_objects=None):
        "Implements Figure 2 of Relation Network for Object Detection"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask[:, None, None, :]
        nbatches = input_query.size(0)

        # tensor with entries R_mn given by a hardcoded embedding of the relative position between bbox_m and bbox_n
        relative_geometry_embeddings = BoxRelationalEmbedding(input_box,
                                                              trignometric_embedding=self.trignometric_embedding)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, self.dim_g)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (input_query, input_key, input_value))]
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
                                              self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.box_attn = box_attention(query, key, value, relative_geometry_weights, mask=mask,
                                         dropout=self.dropout, not_objects=not_objects)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        # An extra internal skip connection is added. This is only
        # kept here for compatibility with some legacy models. In
        # general, there is no advantage in using it, as there is
        # already an outer skip connection surrounding this layer.
        if self.legacy_extra_skip:
            x = input_value + x

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RelationTransformerModel(CaptionModel):

    def make_model(self, tgt_vocab, encoder_N=6, decoder_N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1, drop_paths=[0.1],
                   trignometric_embedding=True, legacy_extra_skip=False, attn_type='box',
                   normalize_out=False):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        if attn_type == 'box':
            attn = BoxMultiHeadedAttention(h, d_model, trignometric_embedding, legacy_extra_skip)
        elif attn_type == 'self':
            attn = MultiHeadedAttention(h, d_model)
        else:
            raise Exception
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        self_attn = MultiHeadedAttention(h, d_model)
        d_layer = DecoderLayer(d_model, None, c(self_attn), c(ff), dropout, drop_paths[0])
        if decoder_N != 0:
            decoder = Decoder(d_layer, decoder_N)
            tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
        else:
            decoder = None
            tgt_embed = None

        kwargs = dict()
        encoder_layers = [EncoderLayer(d_model, c(attn), attn_type, c(ff), dropout, drop_paths[i])
                          for i in range(encoder_N)]
        model = EncoderDecoder(
            Encoder(encoder_layers),
            decoder,
            lambda x: x,  # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            tgt_embed,  # nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            # Generator(d_model * 2 if self.encoder_type == 'mix' else d_model, stage_outputs, normalize_out),
            **kwargs)

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model, position

    def __init__(self, opt, encoder_type='mix'):
        super(RelationTransformerModel, self).__init__()
        self.opt = opt
        # d_model = self.input_encoding_size # 512

        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.att_feat_size = opt.att_feat_size
        self.encoder_type = encoder_type

        self.use_bn = getattr(opt, 'use_bn', 0)
        self.frame_feature = getattr(opt, 'use_frame_features', False)

        self.ss_prob = 0.0  # Schedule sampling probability

        self.obj_feat_block = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.input_encoding_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ())))

        # self.frame_attention = nn.ModuleList([copy.deepcopy(frame_attention) for _ in range(opt.encoder_num_layers)])
        frame_attention = ConvFlattenLinear(opt.input_encoding_size, self.drop_prob_lm)
        self.clip_feat_block = frame_attention

        self.cls_embedding_obj = nn.Embedding(1, opt.input_encoding_size)
        self.cls_embedding_background = nn.Embedding(1, opt.input_encoding_size)

        if hasattr(opt, 'num_multi_frames'):
            self.segment_embed = nn.Embedding(opt.num_multi_frames, self.input_encoding_size)

        self.box_trignometric_embedding = getattr(opt, 'box_trignometric_embedding', True)
        self.legacy_extra_skip = getattr(opt, 'legacy_extra_skip', False)

        tgt_vocab = self.vocab_size + 1
        dpr = [x.item() for x in torch.linspace(0, opt.drop_path, opt.encoder_num_layers*2)]

        if encoder_type == 'object_only' or encoder_type == 'mix':
            self.obj_model, self.position = self.make_model(
                tgt_vocab, encoder_N=opt.encoder_num_layers, decoder_N=opt.decoder_num_layers,
                d_model=opt.input_encoding_size, d_ff=opt.rnn_size,
                trignometric_embedding=self.box_trignometric_embedding,
                legacy_extra_skip=self.legacy_extra_skip, attn_type='box',
                normalize_out=opt.normalize_out, dropout=opt.dropout, drop_paths=dpr[:opt.encoder_num_layers])

        if encoder_type == 'i3d_only' or encoder_type == 'mix':
            self.action_model, self.position = self.make_model(
                tgt_vocab, encoder_N=opt.encoder_num_layers, decoder_N=opt.decoder_num_layers,
                d_model=opt.input_encoding_size, d_ff=opt.rnn_size,
                trignometric_embedding=self.box_trignometric_embedding,
                legacy_extra_skip=self.legacy_extra_skip, attn_type='self',
                normalize_out=opt.normalize_out, dropout=opt.dropout, drop_paths=dpr[:opt.encoder_num_layers]
            )
        if encoder_type == 'mix':
            self.obj_model_2, _ = self.make_model(
                tgt_vocab, encoder_N=opt.encoder_num_layers, decoder_N=opt.decoder_num_layers,
                d_model=opt.input_encoding_size, d_ff=opt.rnn_size,
                trignometric_embedding=self.box_trignometric_embedding,
                legacy_extra_skip=self.legacy_extra_skip, attn_type='self',
                normalize_out=opt.normalize_out, dropout=opt.dropout, drop_paths=dpr[opt.encoder_num_layers:]
            )

            self.action_model_2, _ = self.make_model(
                tgt_vocab, encoder_N=opt.encoder_num_layers, decoder_N=opt.decoder_num_layers,
                d_model=opt.input_encoding_size, d_ff=opt.rnn_size,
                trignometric_embedding=self.box_trignometric_embedding,
                legacy_extra_skip=self.legacy_extra_skip, attn_type='self',
                normalize_out=opt.normalize_out, dropout=opt.dropout, drop_paths=dpr[opt.encoder_num_layers:]
            )

    # def init_hidden(self, bsz):
    #     weight = next(self.parameters())
    #     return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
    #             weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def prepare_feature(self, att_feats, att_masks=None,
                        boxes=None, seq=None, segment_labels=None):

        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.obj_feat_block, att_feats, att_masks)
        if segment_labels is not None:
            segment_embeddings = self.segment_embed(segment_labels)
            att_feats += segment_embeddings
        boxes = self.clip_att(boxes, att_masks)[0]

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] = 1

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, boxes, seq, att_masks, seq_mask

    def _sample_feature(self, obj_features, obj_boxes,
                        action_features=None, not_objects=None):

        if self.encoder_type == 'object_only':
            obj_memory = self.obj_model.encode(obj_features, obj_boxes, src_mask=None, not_objects=not_objects)
            final_memory = obj_memory
        elif self.encoder_type == 'mix':
            # # if all box coords are zero, the box isn't exist
            # obj_mask = (obj_boxes == 0).all(dim=2).logical_not()
            # obj_mask[:, 0] = True
            obj_memory = self.obj_model.encode(obj_features, obj_boxes, src_mask=None, not_objects=not_objects)
            action_memory = self.action_model.encode(action_features, None, src_mask=None)
            action_mem_final = self.action_model_2.encode(action_memory, None, src_mask=None, memory=obj_memory)
            obj_mem_final = self.obj_model_2.encode(obj_memory, None, src_mask=None, memory=action_memory)
            final_memory = torch.cat((action_mem_final[:, 0], obj_mem_final[:, 0]), dim=1).unsqueeze(1)
        elif self.encoder_type == 'i3d_only':
            action_memory = self.action_model.encode(action_features, None, src_mask=None)
            final_memory = action_memory

        return final_memory


class ConvFlattenLinear(nn.Module):
    def __init__(self, input_encoding_size, drop_prob_lm):
        super(ConvFlattenLinear, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=input_encoding_size, kernel_size=1, padding=0, bias=False)
        # self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_prob_lm)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # x = self.conv2(x)
        # x = self.relu(x)
        x = x.flatten(2, -1).transpose(1, 2)
        x = self.dropout(x)
        return x
