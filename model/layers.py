import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, hidden_dim, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))  
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), hidden_dim=self.d_v)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn


class GlobalGate(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head = 1
        self.self_attention = MultiHeadAttention(self.head, self.hidden_dim, self.hidden_dim//self.head, self.hidden_dim//self.head)
        self.self_attention2 = MultiHeadAttention(self.head, self.hidden_dim, self.hidden_dim//self.head, self.hidden_dim//self.head)
        self.G2Gupdategate = nn.Linear(2*self.hidden_dim, self.hidden_dim, bias=True)  

    def forward(self, layer_output, global_matrix=None):

        if global_matrix is not None:
            layer_output_selfatten, _ = self.self_attention(layer_output, layer_output, layer_output) #(b,l,h)
            input_cat = torch.cat([layer_output_selfatten, global_matrix], dim=2) #(b,l,2h)
            update_gate = torch.sigmoid(self.G2Gupdategate(input_cat))
            new_global_matrix = update_gate * layer_output_selfatten + (1-update_gate) * global_matrix

        else:
            new_global_matrix, _ = self.self_attention(layer_output, layer_output, layer_output)

        return new_global_matrix


class LayerGate(nn.Module):

    def __init__(self, hidden_dim, input_dim, use_gaz=True ,gpu=True):
        super().__init__()
        self.hidden_dim = hidden_dim   # layer hidden dim
        self.input_dim = input_dim  # input gaz embed dim
        self.use_gaz = use_gaz

        self.index = torch.LongTensor([[i+j*hidden_dim*4 for i in range(self.hidden_dim*4)] for j in range(4) ])  #(4,H) [[0:H],[h:2H],[2H:3H],[3H:4H]]  H=4h
        self.index2 = torch.LongTensor([[i+j*hidden_dim*3 for i in range(self.hidden_dim*3)] for j in range(4) ])  #(4,H) H=3h
        if gpu:
            self.index = self.index.cuda()
            self.index2 = self.index2.cuda()

        self.cat2gates = nn.Linear(self.hidden_dim*2+self.input_dim, self.hidden_dim*4 *4)  #para run
        self.exper2gates = nn.Linear(self.hidden_dim, self.hidden_dim*3 *4)
        self.reset_paras()


    def reset_paras(self,):
        stdv = 1. / math.sqrt(self.cat2gates.weight.size(1))
        stdv2 = 1. / math.sqrt(self.exper2gates.weight.size(1))

        for layer in range(4):
            nn.init.constant_(self.cat2gates.bias[self.hidden_dim*4*layer:self.hidden_dim*4*(layer+1)].data, val=0)
            nn.init.constant_(self.exper2gates.bias[self.hidden_dim*3*layer:self.hidden_dim*3*(layer+1)].data, val=0)
            for i in range(4):
                start = layer*4 + i
                nn.init.xavier_normal_(self.cat2gates.weight[self.hidden_dim*start:self.hidden_dim*(start+1),:])
            for i in range(3):
                start = layer*3 + i
                nn.init.xavier_normal_(self.exper2gates.weight[self.hidden_dim*start:self.hidden_dim*(start+1),:])


    def forward(self, CNN_output, gaz_input, gaz_input_back, global_matrix, exper_input=None, gaz_mask=None):
        batch_size = global_matrix.size(0)
        seq_len = global_matrix.size(1)

        index = self.index.unsqueeze(1).repeat(1,seq_len,1)  #(4,l,4h)
        index = index.view(1,-1,self.hidden_dim*4).repeat(batch_size, 1, 1)   #(b,4l,4h)

        index2 = self.index2.unsqueeze(1).repeat(1,seq_len,1)  #(4,l,3h)
        index2 = index2.view(1,-1,self.hidden_dim*3).repeat(batch_size, 1, 1)   #(b,4l,3h)

        if self.use_gaz:
            gaz_input = torch.cat([gaz_input,gaz_input_back,gaz_input,gaz_input_back], dim=1)   #(b,4l,i)

        seq_len_cat = seq_len * 4
        global_matrix = global_matrix.repeat(1,4,1)  #(b,4l,h)

        if exper_input is not None:
            exper_input = exper_input.repeat(1,4,1)   #(b,4l,h)

        if self.use_gaz:
            cat_input = torch.cat([CNN_output, gaz_input, global_matrix], dim=2)  #(b,4l,2*h+gaz_dim)
        else:
            cat_input = torch.cat([CNN_output, global_matrix], dim=2)  #(b,4l,2*h+gaz_dim)

        cat_gates_ = self.cat2gates(cat_input)  #(b,4l,4h*4)
        cat_gates = torch.gather(cat_gates_, dim=-1, index=index)  #(b,4l,4h)

        new_state = torch.tanh(cat_gates[:,:,:self.hidden_dim])  #(b,4l,h)
        gates = cat_gates[:,:,self.hidden_dim:]   #(b,4l,3h)

        if exper_input is not None:
            exper_gates_ = self.exper2gates(exper_input)  #(b,l,3h)
            exper_gates = torch.gather(exper_gates_, dim=-1, index=index2)

            gates = gates + exper_gates  #(b,4l,3h)

            state_cat = torch.cat([new_state.unsqueeze(2), CNN_output.unsqueeze(2),exper_input.unsqueeze(2)],dim=2)
        else:
            gates = gates[:,:,:self.hidden_dim*2]  #(b,l,2h)

            state_cat = torch.cat([new_state.unsqueeze(2), CNN_output.unsqueeze(2)],dim=2)

        gates = torch.sigmoid(gates)
        gates = F.softmax(gates.view(batch_size, seq_len_cat, -1, self.hidden_dim),dim=2)

        layer_output = torch.sum(F.mul(gates, state_cat),dim=2 )  #(b,4l,h)

        output = torch.split(layer_output,seq_len,dim=1)

        return output


class MultiscaleAttention(nn.Module):
    
    def __init__(self, num_layer, dropout):
        super().__init__()
        self.MLP_layer = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(num_layer, num_layer),nn.Tanh(), nn.Dropout(p=dropout), nn.Linear(num_layer, num_layer),nn.Tanh())
        
    def forward(self, X_list):
        seq_len = X_list.size(1)

        X_sum = torch.sum(X_list,dim=-1)   #(batch_size,seq_len,num_layer)
        MLP_output = self.MLP_layer(X_sum)
        weights = F.softmax(MLP_output,dim=-1)   #(b,m,l)
        weights_k = weights.unsqueeze(2)  #(b,m,1,l)

        weights_k = weights_k.view(-1,weights_k.size()[2],weights_k.size()[3])
        X_list_ = X_list.view(-1,X_list.size()[2],X_list.size()[3])
        X_attention = torch.bmm(weights_k, X_list_).squeeze(1)   #(b*m,1,l)*(b*m,l,k) = (b*m,1,k) ->(b*m,k)
        X_attention = X_attention.view(-1,seq_len,X_attention.size()[-1])  #(b,m,k)
        
        return X_attention
