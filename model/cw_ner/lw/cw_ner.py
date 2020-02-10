__author__ = "liuwei"

"""
A char word model
"""
import numpy as np
import torch
import torch.nn as nn

from model.cw_ner.modules.gaz_embed import Gaz_Embed
from model.cw_ner.modules.gaz_bilstm import Gaz_BiLSTM
from model.cw_ner.model.crf import CRF
from model.cw_ner.functions.utils import random_embedding
from model.cw_ner.functions.gaz_opt import get_batch_gaz
from model.cw_ner.functions.utils import reverse_padded_sequence


class CW_NER(torch.nn.Module):
    def __init__(self, data, type=1):
        print("Build char-word based NER Task...")
        super(CW_NER, self).__init__()

        self.gpu = data.HP_gpu
        self.label_size = data.label_alphabet_size
        self.type = type
        self.gaz_embed = Gaz_Embed(data, type)

        self.char_embedding = nn.Embedding(data.char_alphabet.size(), data.char_emb_dim)

        self.lstm = Gaz_BiLSTM(data, data.char_emb_dim + data.gaz_emb_dim, data.HP_hidden_dim)

        self.crf = CRF(data.label_alphabet_size, self.gpu)

        self.hidden2tag = nn.Linear(data.HP_hidden_dim * 2, data.label_alphabet_size + 2)

        if data.pretrain_char_embedding is not None:
            self.char_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_char_embedding))
        else:
            self.char_embedding.weight.data.copy_(
                random_embedding(data.char_alphabet_size, data.char_emb_dim)
            )

        if self.gpu:
            self.char_embedding = self.char_embedding.cuda()
            self.hidden2tag = self.hidden2tag.cuda()

    def neg_log_likelihood_loss(self, gaz_list, reverse_gaz_list, char_inputs, char_seq_lengths, batch_label, mask):
        """
        get the neg_log_likelihood_loss
        Args:
            gaz_list: the batch data's gaz, for every chinese char
            reverse_gaz_list: the reverse list
            char_inputs: word input ids, [batch_size, seq_len]
            char_seq_lengths: [batch_size]
            batch_label: [batch_size, seq_len]
            mask: [batch_size, seq_len]
        """
        batch_size = char_inputs.size(0)
        seq_len = char_inputs.size(1)
        lengths = list(map(int, char_seq_lengths))

        # print('one ', reverse_gaz_list[0][:10])

        # get batch gaz ids
        batch_gaz_ids, batch_gaz_length, batch_gaz_mask = get_batch_gaz(reverse_gaz_list, batch_size, seq_len, self.gpu)

        # print('two ', batch_gaz_ids[0][:10])

        reverse_batch_gaz_ids, reverse_batch_gaz_length, reverse_batch_gaz_mask = get_batch_gaz(gaz_list, batch_size, seq_len, self.gpu)
        reverse_batch_gaz_ids = reverse_padded_sequence(reverse_batch_gaz_ids, lengths)
        reverse_batch_gaz_length = reverse_padded_sequence(reverse_batch_gaz_length, lengths)
        reverse_batch_gaz_mask = reverse_padded_sequence(reverse_batch_gaz_mask, lengths)

        # word embedding (32,117,50)
        char_embs = self.char_embedding(char_inputs)
        reverse_char_embs = reverse_padded_sequence(char_embs, lengths)

        # gaz embedding (32,117,50)
        gaz_embs = self.gaz_embed((batch_gaz_ids, batch_gaz_length, batch_gaz_mask))
        reverse_gaz_embs = self.gaz_embed((reverse_batch_gaz_ids, reverse_batch_gaz_length, reverse_batch_gaz_mask))
        # print(gaz_embs[0][0][:20])

        # lstm (32,117,50*2)
        forward_inputs = torch.cat((char_embs, gaz_embs), dim=-1)
        backward_inputs = torch.cat((reverse_char_embs, reverse_gaz_embs), dim=-1)

        lstm_outs, _ = self.lstm((forward_inputs, backward_inputs), char_seq_lengths)

        # hidden2tag
        outs = self.hidden2tag(lstm_outs)

        # crf and loss
        loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        _, tag_seq = self.crf._viterbi_decode(outs, mask)

        return loss, tag_seq

    def forward(self, gaz_list, reverse_gaz_list, char_inputs, char_seq_lengths, mask):
        """
        Args:
            gaz_list: the forward gaz_list
            reverse_gaz_list: the backward gaz list
            char_inputs: word ids
            char_seq_lengths: each sentence length
            mask: sentence mask
        """
        batch_size = char_inputs.size(0)
        seq_len = char_inputs.size(1)
        lengths = list(map(int, char_seq_lengths))

        # get batch gaz ids
        batch_gaz_ids, batch_gaz_length, batch_gaz_mask = get_batch_gaz(reverse_gaz_list, batch_size, seq_len, self.gpu)
        reverse_batch_gaz_ids, reverse_batch_gaz_length, reverse_batch_gaz_mask = get_batch_gaz(gaz_list, batch_size, seq_len, self.gpu)
        reverse_batch_gaz_ids = reverse_padded_sequence(reverse_batch_gaz_ids, lengths)
        reverse_batch_gaz_length = reverse_padded_sequence(reverse_batch_gaz_length, lengths)
        reverse_batch_gaz_mask = reverse_padded_sequence(reverse_batch_gaz_mask, lengths)

        # word embedding
        char_embs = self.char_embedding(char_inputs)
        reverse_char_embs = reverse_padded_sequence(char_embs, lengths)

        # gaz embedding
        gaz_embs = self.gaz_embed((batch_gaz_ids, batch_gaz_length, batch_gaz_mask))
        reverse_gaz_embs = self.gaz_embed((reverse_batch_gaz_ids, reverse_batch_gaz_length, reverse_batch_gaz_mask))

        # lstm
        forward_inputs = torch.cat((char_embs, gaz_embs), dim=-1)
        backward_inputs = torch.cat((reverse_char_embs, reverse_gaz_embs), dim=-1)

        lstm_outs, _ = self.lstm((forward_inputs, backward_inputs), char_seq_lengths)

        # hidden2tag
        outs = self.hidden2tag(lstm_outs)

        # crf and loss
        _, tag_seq = self.crf._viterbi_decode(outs, mask)

        return tag_seq
