# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

from model.crf import CRF
from model.layers import GlobalGate, LayerGate, MultiscaleAttention


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class CNNmodel(nn.Module):
    def __init__(self, data):
        super(CNNmodel, self).__init__()
        self.gpu = data.HP_gpu
        self.use_biword = data.use_bichar
        self.use_posi = data.HP_use_posi
        self.hidden_dim = data.HP_hidden_dim
        self.gaz_alphabet = data.gaz_alphabet
        self.gaz_emb_dim = data.gaz_emb_dim
        self.word_emb_dim = data.char_emb_dim
        self.posi_emb_dim = data.posi_emb_dim
        self.biword_emb_dim = data.bichar_emb_dim
        self.rethink_iter = data.HP_rethink_iter

        scale = np.sqrt(3.0 / self.gaz_emb_dim)
        data.pretrain_gaz_embedding[0, :] = np.random.uniform(-scale, scale, [1, self.gaz_emb_dim])
        self.gaz_embedding = nn.Embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)
        self.gaz_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_gaz_embedding))

        self.word_embedding = nn.Embedding(data.char_alphabet.size(), self.word_emb_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_char_embedding))

        if data.HP_use_posi:
            data.posi_alphabet_size += 1
            self.position_embedding = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(data.posi_alphabet_size, self.posi_emb_dim), freeze=True)

        if self.use_biword:
            self.biword_embedding = nn.Embedding(data.bichar_alphabet.size(), self.biword_emb_dim)
            self.biword_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_bichar_embedding))

        self.drop = nn.Dropout(p=data.HP_dropout)
        self.num_layer = data.HP_num_layer

        input_dim = self.word_emb_dim
        if self.use_biword:
            input_dim += self.biword_emb_dim
        if self.use_posi:
            input_dim += self.posi_emb_dim

        self.cnn_layer0 = nn.Conv1d(input_dim, self.hidden_dim, kernel_size=1, padding=0)
        self.cnn_layers = [nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=2, padding=0) for i in range(self.num_layer - 1)]
        self.cnn_layers_back = [nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=2, padding=0) for i in range(self.num_layer - 1)]
        self.res_cnn_layers = [nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=i + 2, padding=0) for i in range(1, self.num_layer - 1)]
        self.res_cnn_layers_back = [nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=i + 2, padding=0) for i in range(1, self.num_layer - 1)]

        self.layer_gate = LayerGate(self.hidden_dim, self.gaz_emb_dim, gpu=self.gpu)
        self.global_gate = GlobalGate(self.hidden_dim)
        self.exper2gate = nn.Linear(self.hidden_dim, self.hidden_dim * 4)
        self.multiscale_layer = MultiscaleAttention(self.num_layer, data.HP_dropout)

        self.hidden2tag = nn.Linear(self.hidden_dim, data.label_alphabet_size + 2)
        self.crf = CRF(data.label_alphabet_size, self.gpu)

        if self.gpu:
            self.gaz_embedding = self.gaz_embedding.cuda()
            self.word_embedding = self.word_embedding.cuda()
            if self.use_posi:
                self.position_embedding = self.position_embedding.cuda()
            if self.use_biword:
                self.biword_embedding = self.biword_embedding.cuda()
            self.cnn_layer0 = self.cnn_layer0.cuda()
            self.multiscale_layer = self.multiscale_layer.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.layer_gate = self.layer_gate.cuda()
            self.global_gate = self.global_gate.cuda()
            self.crf = self.crf.cuda()
            for i in range(self.num_layer - 1):
                self.cnn_layers[i] = self.cnn_layers[i].cuda()
                self.cnn_layers_back[i] = self.cnn_layers_back[i].cuda()
                if i >= 1:
                    self.res_cnn_layers[i - 1] = self.res_cnn_layers[i - 1].cuda()
                    self.res_cnn_layers_back[i - 1] = self.res_cnn_layers_back[i - 1].cuda()

    def get_tags(self, gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_mask_input, mask):

        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)

        word_embs = self.word_embedding(word_inputs)

        if self.use_biword:
            biword_embs = self.biword_embedding(biword_inputs)
            word_embs = torch.cat([word_embs, biword_embs], dim=2)

        if self.use_posi:
            posi_inputs = torch.zeros(batch_size, seq_len).long()

            posi_inputs[:, :] = torch.LongTensor([i + 1 for i in range(seq_len)])
            if self.gpu:
                posi_inputs = posi_inputs.cuda()
            position_embs = self.position_embedding(posi_inputs)
            word_embs = torch.cat([word_embs, position_embs], dim=2)

        word_inputs_d = self.drop(word_embs)
        word_inputs_d = word_inputs_d.transpose(2, 1).contiguous()

        X_pre = self.cnn_layer0(word_inputs_d)  # (batch_size,hidden_size,seq_len)
        X_pre = torch.tanh(X_pre)

        X_trans = X_pre.transpose(2, 1).contiguous()

        global_matrix0 = self.global_gate(X_trans)  # G0

        X_list = X_trans.unsqueeze(2)  # (batch_size,seq_len,num_layer,hidden_size)

        padding = torch.zeros(batch_size, self.hidden_dim, 1)
        if self.gpu:
            padding = padding.cuda()

        feed_back = None
        for iteration in range(self.rethink_iter):

            global_matrix = global_matrix0
            X_pre = self.drop(X_pre)

            X_pre_padding = torch.cat([X_pre, padding], dim=2)  # (batch_size,hidden_size,seq_len+1)

            X_pre_padding_back = torch.cat([padding, X_pre], dim=2)

            for layer in range(self.num_layer - 1):

                X = self.cnn_layers[layer](X_pre_padding)  # X: (batch_size,hidden_size,seq_len)
                X = torch.tanh(X)

                X_back = self.cnn_layers_back[layer](X_pre_padding_back)  # X: (batch_size,hidden_size,seq_len)
                X_back = torch.tanh(X_back)

                if layer > 0:
                    windowpad = torch.cat([padding for i in range(layer)], dim=2)
                    X_pre_padding_w = torch.cat([X_pre, windowpad, padding], dim=2)
                    X_res = self.res_cnn_layers[layer - 1](X_pre_padding_w)
                    X_res = torch.tanh(X_res)

                    X_pre_padding_w_back = torch.cat([padding, windowpad, X_pre], dim=2)
                    X_res_back = self.res_cnn_layers_back[layer - 1](X_pre_padding_w_back)
                    X_res_back = torch.tanh(X_res_back)

                layer_gaz_back = torch.zeros(batch_size, seq_len).long()

                if seq_len > layer + 1:
                    layer_gaz_back[:, layer + 1:] = layer_gaz[:, :seq_len - layer - 1, layer]

                if self.gpu:
                    layer_gaz_back = layer_gaz_back.cuda()

                gazs_embeds = self.gaz_embedding(layer_gaz[:, :, layer])
                gazs_embeds_back = self.gaz_embedding(layer_gaz_back)

                mask_gaz = (mask == 0).unsqueeze(-1).repeat(1, 1, self.gaz_emb_dim)
                gazs_embeds = gazs_embeds.masked_fill(mask_gaz, 0)
                gazs_embeds_back = gazs_embeds_back.masked_fill(mask_gaz, 0)

                gazs_embeds = self.drop(gazs_embeds)
                gazs_embeds_back = self.drop(gazs_embeds_back)

                if layer > 0:  # res
                    X_input = torch.cat([X, X_back, X_res, X_res_back], dim=-1).transpose(2, 1).contiguous()  # (b,4l,h)
                    X, X_back, X_res, X_res_back = self.layer_gate(X_input, gazs_embeds, gazs_embeds_back, global_matrix, exper_input=feed_back, gaz_mask=None)
                    X = X + X_back + X_res + X_res_back
                else:
                    X_input = torch.cat([X, X_back, X, X_back], dim=-1).transpose(2, 1).contiguous()  # (b,4l,h)
                    X, X_back, _, _ = self.layer_gate(X_input, gazs_embeds, gazs_embeds_back, global_matrix, exper_input=feed_back, gaz_mask=None)
                    X = X + X_back

                global_matrix = self.global_gate(X, global_matrix)
                if iteration == self.rethink_iter - 1:
                    X_list = torch.cat([X_list, X.unsqueeze(2)], dim=2)
                if layer == self.num_layer - 2:
                    feed_back = X

                X = X.transpose(2, 1).contiguous()
                X_d = self.drop(X)

                X_pre_padding = torch.cat([X_d, padding], dim=2)  # padding

                padding_back = torch.cat([padding for _ in range(min(layer + 2, seq_len + 1))], dim=2)
                if seq_len > layer + 1:
                    X_pre_padding_back = torch.cat([padding_back, X_d[:, :, :seq_len - layer - 1]], dim=2)  # (b,h,seqlen+1)
                else:
                    X_pre_padding_back = padding_back

        X_attention = self.multiscale_layer(X_list)
        tags = self.hidden2tag(X_attention)  # (b,l,t)

        return tags

    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, layer_gaz, gaz_mask, mask, batch_label):

        tags = self.get_tags(gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_mask, mask)

        total_loss = self.crf.neg_log_likelihood_loss(tags, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(tags, mask)

        return total_loss, tag_seq

    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, layer_gaz, gaz_mask, mask):

        tags = self.get_tags(gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_mask, mask)

        scores, tag_seq = self.crf._viterbi_decode(tags, mask)

        return tag_seq
