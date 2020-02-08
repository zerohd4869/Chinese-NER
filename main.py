import argparse
import copy
import gc
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
from tqdm import tqdm

import load_conf
from model.bilstmcrf import BiLSTM_CRF
from model.CNNmodel import CNNmodel
from utils.data import Data
from utils.metric import get_ner_fmeasure

seed_num = 2019
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def data_initialization(data, gaz_file, train_file, dev_file, test_file):
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    data.build_gaz_file(gaz_file)

    # build gaz_alphabet by Word EnumerateMatchList
    data.build_gaz_alphabet(train_file)
    data.build_gaz_alphabet(dev_file)
    data.build_gaz_alphabet(test_file)
    data.fix_alphabet()
    return data


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold  result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label_2(pred_variable, gold_variable, mask_variable, label_alphabet):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(int(pred_tag[idx][idy])) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)

    return pred_label, gold_label


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        # print "p:",pred, pred_tag.tolist()
        # print "g:", gold, gold_tag.tolist()
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


# class MacOSFile(object):
#     def __init__(self, f):
#         self.f = f

#     def __getattr__(self, item):
#         return getattr(self.f, item)

#     def read(self, n):
#         if n >= (1 << 31):
#             buffer = bytearray(n)
#             pos = 0
#             while pos < n:
#                 size = min(n - pos, 1 << 31 - 1)
#                 chunk = self.f.read(size)
#                 buffer[pos:pos + size] = chunk
#                 pos += size
#             return buffer
#         return self.f.read(n)


def save_data_setting(data, save_file):
    new_data = copy.deepcopy(data)
    # remove input instances
    new_data.train_texts = []
    new_data.dev_texts = []
    new_data.test_texts = []
    new_data.raw_texts = []

    new_data.train_Ids = []
    new_data.dev_Ids = []
    new_data.test_Ids = []
    new_data.raw_Ids = []
    # save data settings

    # if not os.path.exists(save_file):
    #     os.makedirs(save_file)

    with open(save_file, 'wb') as fp:
        # pickle.dump(obj, file[, protocol]) 
        # 序列化对象，并将结果数据流写入到文件对象中
        # 参数protocol是序列化模式，默认值为0，表示以文本的形式序列化。protocol的值还可以是1或2，表示以二进制的形式序列化
        # pickle.load(file) 反序列化对象。将文件中的数据解析为一个Python对象.要让python能够找到类的定义
        # 若Pickle的对象太大，超过了2G，在OSX系统中无法直接dump
        pickle.dump(new_data, fp)
    print("Data setting saved to file: ", save_file)


def load_data_setting(save_file):
    with open(save_file, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", save_file)
    data.show_data_summary()
    return data


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    print("Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def evaluate(data, model, name):
    global instances
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
    pred_results = []
    gold_results = []
    # set model in eval model
    model.eval()
    batch_size = 10
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        pred_label, gold_label = -1, -1
        if data.model_name == 'CNN_model':
            gaz_list, batch_char, batch_bichar, batch_charlen, batch_label, layer_gaz, gaz_mask, mask = batchify_with_label_2(instance, data.HP_gpu,
                                                                                                                              data.HP_num_layer, True)
            tag_seq = model(gaz_list, batch_char, batch_bichar, batch_charlen, layer_gaz, gaz_mask, mask)

            pred_label, gold_label = recover_label_2(tag_seq, batch_label, mask, data.label_alphabet)
        elif data.model_name == 'LSTM_model':
            gaz_list, batch_char, batch_bichar, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(instance, data.HP_gpu, True)
            tag_seq = model(gaz_list, batch_char, batch_bichar, batch_charlen, mask)
            pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_charrecover)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances) / decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    return speed, acc, p, r, f, pred_results


def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,biwords,chars,gaz, labels],[words,biwords,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len) 
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            char_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            character_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order 
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len) 
    """
    batch_size = len(input_batch_list)
    chars = [sent[0] for sent in input_batch_list]
    bichars = [sent[1] for sent in input_batch_list]
    # chars = [sent[2] for sent in input_batch_list]

    gazs = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    char_seq_lengths = torch.LongTensor(list(map(len, chars)))
    max_seq_len = char_seq_lengths.max().item()

    with torch.no_grad():
        # torch.zeros(*sizes, out=None) → Tensor
        # 返回一个全为标量 0 的张量，形状由可变参数sizes 定义
        # sizes (int...) – 整数序列，定义了输出形状
        # out(Tensor, optional) – 结果张量
        char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
        bichar_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
        label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
        mask = autograd.Variable(torch.zeros((batch_size, max_seq_len))).byte()

    for idx, (seq, biseq, label, seqlen) in enumerate(zip(chars, bichars, labels, char_seq_lengths)):
        # torch.Tensor是一种包含单一数据类型元素的多维矩阵
        # 64-bit integer (signed)	torch.LongTensor	torch.cuda.LongTensor
        char_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        bichar_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen.item())

    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    bichar_seq_tensor = bichar_seq_tensor[char_perm_idx]
    label_seq_tensor = label_seq_tensor[char_perm_idx]
    mask = mask[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)

    # keep the gaz_list in orignial order
    gaz_list = [gazs[i] for i in char_perm_idx]
    gaz_list.append(volatile_flag)
    if gpu:
        char_seq_tensor = char_seq_tensor.cuda()
        bichar_seq_tensor = bichar_seq_tensor.cuda()
        char_seq_lengths = char_seq_lengths.cuda()
        char_seq_recover = char_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()

        mask = mask.cuda()
    return gaz_list, char_seq_tensor, bichar_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def batchify_with_label_2(input_batch_list, gpu, num_layer, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,biwords,chars,gaz, labels],[words,biwords,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            char_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            character_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    chars = [sent[0] for sent in input_batch_list]
    bichars = [sent[1] for sent in input_batch_list]
    # chars = [sent[2] for sent in input_batch_list]

    gazs = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]

    layer_gazs = [sent[4] for sent in input_batch_list]
    gaz_mask = [sent[5] for sent in input_batch_list]

    char_seq_lengths = torch.LongTensor(list(map(len, chars)))
    max_seq_len = char_seq_lengths.max().item()

    with torch.no_grad():
        # torch.zeros(*sizes, out=None) → Tensor
        # 返回一个全为标量 0 的张量，形状由可变参数sizes 定义
        # sizes (int...) – 整数序列，定义了输出形状
        # out(Tensor, optional) – 结果张量
        char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
        bichar_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
        label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
        mask = autograd.Variable(torch.zeros((batch_size, max_seq_len))).byte()
        gaz_mask_tensor = torch.zeros((batch_size, max_seq_len, num_layer)).byte()
        layer_gaz_tensor = torch.zeros(batch_size, max_seq_len, num_layer).long()

    for idx, (seq, biseq, label, seqlen, layergaz, gazmask) in enumerate(zip(chars, bichars, labels, char_seq_lengths, layer_gazs, gaz_mask)):
        # torch.Tensor是一种包含单一数据类型元素的多维矩阵
        # 64-bit integer (signed)	torch.LongTensor	torch.cuda.LongTensor
        char_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        bichar_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen.item())
        gaz_mask_tensor[idx, :seqlen] = torch.LongTensor(gazmask)
        layer_gaz_tensor[idx, :seqlen] = torch.LongTensor(layergaz)

    # char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    # char_seq_tensor = char_seq_tensor[char_perm_idx]
    # bichar_seq_tensor = bichar_seq_tensor[char_perm_idx]
    # label_seq_tensor = label_seq_tensor[char_perm_idx]
    # mask = mask[char_perm_idx]
    # _, char_seq_recover = char_perm_idx.sort(0, descending=False)

    # keep the gaz_list in orignial order
    # gaz_list = [gazs[i] for i in char_perm_idx]
    # gaz_list.append(volatile_flag)
    if gpu:
        char_seq_tensor = char_seq_tensor.cuda()
        bichar_seq_tensor = bichar_seq_tensor.cuda()
        char_seq_lengths = char_seq_lengths.cuda()
        # char_seq_recover = char_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        layer_gaz_tensor = layer_gaz_tensor.cuda()
        gaz_mask_tensor = gaz_mask_tensor.cuda()
        mask = mask.cuda()
    return gazs, char_seq_tensor, bichar_seq_tensor, char_seq_lengths, label_seq_tensor, layer_gaz_tensor, gaz_mask_tensor, mask


def train(data, save_model_dir, dset_dir, seg=True):
    print("Training model...")
    data.show_data_summary()
    save_data_setting(data, dset_dir)
    model = None
    if data.model_name == 'CNN_model':
        model = CNNmodel(data)
    elif data.model_name == 'LSTM_model':
        model = BiLSTM_CRF(data)
    assert (model is not None)

    print("finished built model.")
    # loss_function = nn.NLLLoss()
    # requires_grad指定要不要更新這個變數 属性默认为False 可以加快運算
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # SGD: Stochastic gradient descent
    # 每读入一个数据，便立刻计算cost fuction的梯度来更新参数
    # 算法收敛速度快 可以在线更新 有几率跳出较差的局部最优
    # 易收敛到局部最优，易被困在鞍点
    # 更新方向完全依赖于当前batch计算出的梯度，因而十分不稳定
    #
    # SGD+momentum
    # 更新的时候在一定程度上保留之前更新的方向，同时利用当前batch的梯度微调最终的更新方向
    # 在一定程度上增加稳定性，从而学习地更快，并且还有一定摆脱局部最优的能力
    # optim: SGD/Adagrad/AdaDelta/RMSprop/Adam. optimizer selection.
    # optimizer = optim.SGD(parameters, lr=data.HP_lr, momentum=data.HP_momentum)
    optimizer = optim.Adagrad(parameters, lr=data.HP_lr)
    best_dev = -1

    # training
    for idx in tqdm(range(data.HP_iteration)):
        epoch_start = time.time()
        print("\nEpoch: %s/%s" % (idx, data.HP_iteration))
        optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        batch_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            tag_seq, batch_label, mask, loss = None, None, None, None
            if data.model_name == 'CNN_model':
                gaz_list, batch_word, batch_biword, batch_wordlen, batch_label, layer_gaz, gaz_mask, mask = batchify_with_label_2(instance, data.HP_gpu,
                                                                                                                                  data.HP_num_layer)
                instance_count += 1
                loss, tag_seq = model.neg_log_likelihood_loss(gaz_list, batch_word, batch_biword, batch_wordlen, layer_gaz, gaz_mask, mask, batch_label)
            elif data.model_name == 'LSTM_model':
                gaz_list, batch_char, batch_bichar, batch_charlen, batch_wordrecover, batch_label, mask = batchify_with_label(instance, data.HP_gpu,
                                                                                                                              data.HP_num_layer)
                instance_count += 1
                loss, tag_seq = model.neg_log_likelihood_loss(gaz_list, batch_char, batch_bichar, batch_charlen, batch_label, mask)
            assert (loss.size!=torch.Size([]))
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            total_loss += loss.item()
            batch_loss += loss

            batch_loss.backward()
            optimizer.step()
            model.zero_grad()
            batch_loss = 0

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (idx, epoch_cost, train_num / epoch_cost, total_loss))

        speed, acc, p, r, f, _ = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        current_score = f if seg else acc
        if seg:
            print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_cost, speed, acc, p, r, f))
        else:
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc))

        if current_score > best_dev:
            if seg:
                print("Exceed previous best f score:", best_dev)
            else:
                print("Exceed previous best acc score:", best_dev)
            save_model_name = save_model_dir + '-' + str(idx) + '-' + str(round(current_score * 100, 1)) + ".model"
            torch.save(model.state_dict(), save_model_name)
            best_dev = current_score
        speed, acc, p, r, f, _ = evaluate(data, model, "test")
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if seg:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (test_cost, speed, acc, p, r, f))
        else:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed, acc))
        gc.collect()


def load_model_decode(model_dir, data, name, gpu, seg=True):
    data.HP_gpu = gpu
    print("Load Model from file: ", model_dir)

    model = None
    if data.model_name == 'CNN_model':
        model = CNNmodel(data)
    elif data.model_name == 'LSTM_model':
        model = BiLSTM_CRF(data)
    assert (model is not None)
    model.load_state_dict(torch.load(model_dir))

    print("Decode %s data ..." % name)
    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(data, model, name)
    end_time = time.time()
    time_cost = end_time - start_time

    # seg: boolen.
    # If task is segmentation like, tasks with token accuracy evaluation (e.g. POS, CCG) is False;
    # tasks with F-value evaluation(e.g. Word Segmentation, NER, Chunking) is True .
    if seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
            name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" % (name, time_cost, speed, acc))
    return pred_results


if __name__ == '__main__':
    """
    python main.py --conf_path ./lrcnn_ner.conf
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', help='path of configure', default='./lrcnn_ner.conf', required=False)
    args = parser.parse_args()
    #conf_dict = load_conf.load_conf(args.conf_path)
    conf_dict = load_conf.load_conf('./lattice_ner.conf')
    print(conf_dict)

    train_file = conf_dict['train']
    dev_file = conf_dict['dev']
    test_file = conf_dict['test']
    model_dir = conf_dict['load_model']
    dset_dir = conf_dict['save_dset']
    output_file = conf_dict['output']
    seg = conf_dict['seg']
    status = conf_dict['status']
    model_name = conf_dict['model']
    save_model_dir = conf_dict['save_model']
    gpu = torch.cuda.is_available()
    # Neural word segmentation with rich pretraining  https://github.com/jiesutd/RichWordSegmentor
    char_emb = conf_dict['char_emb']  # "./data/gigaword_chn.all.a2b.uni.ite50.vec"
    bichar_emb = conf_dict['bichar_emb']  # "./data/gigaword_chn.all.a2b.bi.ite50.vec"
    gaz_file = conf_dict['gaz_file']  # "./data/ctb.50d.vec"

    if gaz_file.lower() == 'none':
        gaz_file = None
    if char_emb.lower() == 'none':
        char_emb = None
    if bichar_emb.lower() == 'none':
        bichar_emb = None

    print("Model:", model_name)
    print("GPU available:", gpu)
    print("Status:", status)
    print("Seg: ", seg)
    print("Train file:", train_file)
    print("Dev file:", dev_file)
    print("Test file:", test_file)
    print("Char emb:", char_emb)
    print("Bichar emb:", bichar_emb)
    print("Gaz file:", gaz_file)
    if status == 'train':
        print("Model saved to:", save_model_dir)
    # 立即把stdout缓存内容输出
    sys.stdout.flush()

    if status == 'train':
        data = Data()
        data.model_name = model_name
        data.HP_gpu = gpu
        data.use_bichar = conf_dict['use_bichar']
        data.HP_batch_size = conf_dict['HP_batch_size']  # 1
        data.HP_iteration = conf_dict['HP_iteration']  # 100
        data.HP_lr = conf_dict['HP_lr']  # 0.015
        data.HP_lr_decay = conf_dict['HP_lr_decay']  # 0.5
        data.HP_hidden_dim = conf_dict['HP_hidden_dim']
        data.MAX_SENTENCE_LENGTH = conf_dict['MAX_SENTENCE_LENGTH']
        data_initialization(data, gaz_file, train_file, dev_file, test_file)

        if data.model_name in ['CNN_model', 'LSTM_model']:
            data.generate_instance_with_gaz_2(train_file, 'train')
            data.generate_instance_with_gaz_2(dev_file, 'dev')
            data.generate_instance_with_gaz_2(test_file, 'test')
        else:
            print("model_name is not set!")
            sys.exit(1)
        data.build_char_pretrain_emb(char_emb)
        data.build_bichar_pretrain_emb(bichar_emb)
        data.build_gaz_pretrain_emb(gaz_file)
        train(data, save_model_dir, dset_dir, seg)
    elif status == 'test':
        data = load_data_setting(dset_dir)
        if data.model_name == 'CNN_model':
            data.generate_instance_with_gaz_2(test_file, 'test')
        elif data.model_name == 'LSTM_model':
            data.generate_instance_with_gaz(test_file, 'test')
        load_model_decode(model_dir, data, 'test', gpu, seg)
    else:
        print("Invalid argument! Please use valid argumentguments! (train/test/decode)")
