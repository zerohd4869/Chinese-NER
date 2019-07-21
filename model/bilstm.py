import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.latticelstm import LatticeLSTM


class BiLSTM(nn.Module):
    def __init__(self, data):
        super(BiLSTM, self).__init__()
        print("build batched bilstm...")
        self.use_bichar = data.use_bichar
        self.gpu = data.HP_gpu
        # self.use_char = data.HP_use_character
        self.use_gaz = data.HP_use_gaz
        self.batch_size = data.HP_batch_size
        self.char_hidden_dim = 0
        self.embedding_dim = data.char_emb_dim
        self.hidden_dim = data.HP_hidden_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.char_embeddings = nn.Embedding(data.char_alphabet.size(), self.embedding_dim)
        self.bichar_embeddings = nn.Embedding(data.bichar_alphabet.size(), data.bichar_emb_dim)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        if data.pretrain_char_embedding is not None:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_char_embedding))
        else:
            self.char_embeddings.weight.data.copy_(
                # torch.from_numpy(_): numpy to  torch
                # torch_data.numpy(): torch to numpy
                torch.from_numpy(self.random_embedding(data.char_alphabet.size(), self.embedding_dim)))

        if data.pretrain_bichar_embedding is not None:
            self.bichar_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_bichar_embedding))
        else:
            self.bichar_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.bichar_alphabet.size(), data.bichar_emb_dim)))
        # The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim.
        lstm_hidden = data.HP_hidden_dim // 2 if self.bilstm_flag else data.HP_hidden_dim
        lstm_input = self.embedding_dim + self.char_hidden_dim
        if self.use_bichar:
            lstm_input += data.bichar_emb_dim
        self.forward_lstm = LatticeLSTM(lstm_input, lstm_hidden, data.gaz_dropout, data.gaz_alphabet.size(),
                                        data.gaz_emb_dim, data.pretrain_gaz_embedding, True, data.HP_fix_gaz_emb,
                                        self.gpu)
        if self.bilstm_flag:
            self.backward_lstm = LatticeLSTM(lstm_input, lstm_hidden, data.gaz_dropout, data.gaz_alphabet.size(),
                                             data.gaz_emb_dim, data.pretrain_gaz_embedding, False, data.HP_fix_gaz_emb,
                                             self.gpu)
        # use biLSTM
        # self.lstm = nn.LSTM(lstm_input, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)

        if self.gpu:
            self.drop = self.drop.cuda()
            self.droplstm = self.droplstm.cuda()
            self.char_embeddings = self.char_embeddings.cuda()
            self.bichar_embeddings = self.bichar_embeddings.cuda()
            self.forward_lstm = self.forward_lstm.cuda()
            if self.bilstm_flag:
                self.backward_lstm = self.backward_lstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def get_lstm_features(self, gaz_list, char_inputs, bichar_inputs, char_seq_lengths):
        """
            input:
                char_inputs: (batch_size, sent_len)
                gaz_list:
                char_seq_lengths: list of batch_size, (batch_size,1)
                character_inputs: (batch_size*sent_len, word_length)
                character_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output: 
                Variable(sent_len, batch_size, hidden_dim)
        """
        char_embs = self.char_embeddings(char_inputs)
        if self.use_bichar:
            bichar_embs = self.bichar_embeddings(bichar_inputs)
            char_embs = torch.cat([char_embs, bichar_embs], 2)

        char_embs = self.drop(char_embs)

        hidden = None
        lstm_out, hidden = self.forward_lstm(char_embs, gaz_list, hidden)
        if self.bilstm_flag:
            backward_hidden = None
            backward_lstm_out, backward_hidden = self.backward_lstm(char_embs, gaz_list, backward_hidden)
            lstm_out = torch.cat([lstm_out, backward_lstm_out], 2)
        lstm_out = self.droplstm(lstm_out)
        return lstm_out

    def get_output_score(self, gaz_list, char_inputs, bichar_inputs, char_seq_lengths):
        lstm_out = self.get_lstm_features(gaz_list, char_inputs, bichar_inputs, char_seq_lengths)
        outputs = self.hidden2tag(lstm_out)
        return outputs

    def neg_log_likelihood_loss(self, gaz_list, char_inputs, bichar_inputs, char_seq_lengths, batch_label, mask):
        ## mask is not used
        batch_size = char_inputs.size(0)
        seq_len = char_inputs.size(1)
        total_word = batch_size * seq_len
        loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
        outs = self.get_output_score(gaz_list, char_inputs, bichar_inputs, char_seq_lengths)
        outs = outs.view(total_word, -1)
        score = F.log_softmax(outs, 1)
        loss = loss_function(score, batch_label.view(total_word))
        _, tag_seq = torch.max(score, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        return loss, tag_seq

    def forward(self, gaz_list, char_inputs, bichar_inputs, char_seq_lengths, mask):

        batch_size = char_inputs.size(0)
        seq_len = char_inputs.size(1)
        total_word = batch_size * seq_len
        outs = self.get_output_score(gaz_list, char_inputs, bichar_inputs, char_seq_lengths)
        outs = outs.view(total_word, -1)
        _, tag_seq = torch.max(outs, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        # filter padded position with zero
        decode_seq = mask.long() * tag_seq
        return decode_seq
