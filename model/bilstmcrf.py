import torch.nn as nn
from model.bilstm import BiLSTM
from model.crf import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, data):
        super(BiLSTM_CRF, self).__init__()
        print("build batched lstmcrf...")
        self.gpu = data.HP_gpu
        # For CRF, we need to add extra two label START and END for downlayer lstm, use original label size for CRF
        label_size = data.label_alphabet_size
        data.label_alphabet_size += 2
        self.lstm = BiLSTM(data)
        self.crf = CRF(label_size, self.gpu)

    def neg_log_likelihood_loss(self, gaz_list, char_inputs, bichar_inputs, char_seq_lengths, batch_label, mask):
        outs = self.lstm.get_output_score(gaz_list, char_inputs, bichar_inputs, char_seq_lengths)
        total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        return total_loss, tag_seq

    def forward(self, gaz_list, char_inputs, bichar_inputs, char_seq_lengths, mask):
        outs = self.lstm.get_output_score(gaz_list, char_inputs, bichar_inputs, char_seq_lengths)
        scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        return tag_seq

    def get_lstm_features(self, gaz_list, char_inputs, bichar_inputs, char_seq_lengths):
        return self.lstm.get_lstm_features(gaz_list, char_inputs, bichar_inputs, char_seq_lengths)
