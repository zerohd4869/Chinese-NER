import sys

from utils.alphabet import Alphabet
from utils.functions import *
from utils.gazetteer import Gazetteer

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"


class Data:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_char_emb = True
        self.norm_bichar_emb = True
        self.norm_gaz_emb = False
        self.char_alphabet = Alphabet('char')
        self.bichar_alphabet = Alphabet('bichar')
        self.label_alphabet = Alphabet('label', True)
        self.gaz_lower = False
        self.gaz = Gazetteer(self.gaz_lower)
        self.gaz_alphabet = Alphabet('gaz')
        self.HP_fix_gaz_emb = False
        self.HP_use_gaz = True

        self.tagScheme = "NoSeg"

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []
        self.use_bichar = True
        self.char_emb_dim = 50
        self.bichar_emb_dim = 50
        self.gaz_emb_dim = 50
        self.posi_emb_dim  = 30
        self.gaz_dropout = 0.5
        self.pretrain_char_embedding = None
        self.pretrain_bichar_embedding = None
        self.pretrain_gaz_embedding = None
        self.label_size = 0
        self.char_alphabet_size = 0
        self.bichar_alphabet_size = 0
        self.character_alphabet_size = 0
        self.label_alphabet_size = 0
        # hyper parameters
        self.HP_iteration = 100
        self.HP_batch_size = 128
        # self.HP_char_hidden_dim = 50  # int. Character hidden vector dimension for character sequence layer.
        self.HP_hidden_dim = 200  # int. Char hidden vector dimension for word sequence layer.
        self.HP_dropout = 0.5  # float. Dropout probability.
        self.HP_lstm_layer = 1  # int. LSTM layer number for word sequence layer.
        self.HP_bilstm = True  # boolen. If use bidirection lstm for word seuquence layer.
        self.HP_gpu = False
        # Word level LSTM models (e.g. char LSTM + word LSTM + CRF) would prefer a `lr` around 0.015.
        # Word level CNN models (e.g. char LSTM + word CNN + CRF) would prefer a `lr` around 0.005 and with more iterations.
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05  # float. Learning rate decay rate, only works when optimizer=SGD.
        self.HP_clip = 5.0  # float. Clip the gradient which is larger than the setted number.
        self.HP_momentum = 0  # float. Momentum


        self.HP_use_posi = False
        self.HP_num_layer = 4
        self.HP_rethink_iter = 2
        self.model_name = 'CNNmodel'
        self.posi_alphabet_size = 0

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Tag          scheme: %s" % (self.tagScheme))
        print("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s" % (self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s" % (self.number_normalized))
        print("     Use          bigram: %s" % (self.use_bichar))
        print("     Char  alphabet size: %s" % (self.char_alphabet_size))
        print("     Bichar alphabet size: %s" % (self.bichar_alphabet_size))
        print("     Gaz   alphabet size: %s" % (self.gaz_alphabet.size()))
        print("     Label alphabet size: %s" % (self.label_alphabet_size))
        print("     Word embedding size: %s" % (self.char_emb_dim))
        print("     Bichar embedding size: %s" % (self.bichar_emb_dim))
        print("     Gaz embedding size: %s" % (self.gaz_emb_dim))
        print("     Norm     word   emb: %s" % (self.norm_char_emb))
        print("     Norm     bichar emb: %s" % (self.norm_bichar_emb))
        print("     Norm     gaz    emb: %s" % (self.norm_gaz_emb))
        print("     Norm   gaz  dropout: %s" % (self.gaz_dropout))
        print("     Train instance number: %s" % (len(self.train_texts)))
        print("     Dev   instance number: %s" % (len(self.dev_texts)))
        print("     Test  instance number: %s" % (len(self.test_texts)))
        print("     Raw   instance number: %s" % (len(self.raw_texts)))
        print("     Hyperpara  iteration: %s" % (self.HP_iteration))
        print("     Hyperpara  batch size: %s" % (self.HP_batch_size))
        print("     Hyperpara          lr: %s" % (self.HP_lr))
        print("     Hyperpara    lr_decay: %s" % (self.HP_lr_decay))
        print("     Hyperpara     HP_clip: %s" % (self.HP_clip))
        print("     Hyperpara    momentum: %s" % (self.HP_momentum))
        print("     Hyperpara  hidden_dim: %s" % (self.HP_hidden_dim))
        print("     Hyperpara     dropout: %s" % (self.HP_dropout))
        print("     Hyperpara  lstm_layer: %s" % (self.HP_lstm_layer))
        print("     Hyperpara      bilstm: %s" % (self.HP_bilstm))
        print("     Hyperpara         GPU: %s" % (self.HP_gpu))
        print("     Hyperpara     use_gaz: %s" % (self.HP_use_gaz))
        print("     Hyperpara fix gaz emb: %s" % (self.HP_fix_gaz_emb))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def refresh_label_alphabet(self, input_file):
        old_size = self.label_alphabet_size
        self.label_alphabet.clear(True)
        in_lines = open(input_file, 'r').readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                label = pairs[-1]
                self.label_alphabet.add(label)
        self.label_alphabet_size = self.label_alphabet.size()
        start_s = False
        start_b = False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                start_s = True
            elif "B-" in label.upper():
                start_b = True
        if start_b:
            if start_s:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
        self.fix_alphabet()
        print("Refresh label alphabet finished: old:%s -> new:%s" % (old_size, self.label_alphabet_size))

    # "陈 B-PER"
    def build_alphabet(self, input_file):
        if input_file is None or not os.path.isfile(input_file):
            # print('[' + sys._getframe().f_code.co_name + '] file ' + str(input_file) + "can not be found or is not a file address")
            return
        with codecs.open(input_file, 'r', 'utf-8') as fr:
            in_lines = fr.readlines()  # list

            seqlen = 0
            for idx in range(len(in_lines)):
                line = in_lines[idx]  # '陈 B-PER\n'
                # 行不空 则加入label word bichar char
                if len(line) > 2:
                    # if sequence labeling data format i.e. CoNLL 2003
                    pairs = line.strip().split()  # list ['陈','B-PER']
                    char = pairs[0]  # '陈'
                    if self.number_normalized:  # 数字转0
                        char = normalize_char(char)
                    label = pairs[-1]  # "B-PER"
                    # build feature alphabet
                    self.label_alphabet.add(label)
                    self.char_alphabet.add(char)
                    if idx < len(in_lines) - 1 and len(in_lines[idx + 1]) > 2:
                        bichar = char + in_lines[idx + 1].strip().split()[0]  # 陈元
                    else:
                        bichar = char + NULLKEY
                    self.bichar_alphabet.add(bichar)
                    # for char in char:
                    #     self.character_alphabet.add(char)
                    seqlen += 1
                else:
                    self.posi_alphabet_size = max(seqlen, self.posi_alphabet_size)
                    seqlen = 0
            self.char_alphabet_size = self.char_alphabet.size()
            self.bichar_alphabet_size = self.bichar_alphabet.size()
            self.label_alphabet_size = self.label_alphabet.size()
            start_s = False
            start_b = False
            for label, _ in self.label_alphabet.iteritems():
                if "S-" in label.upper():
                    start_s = True
                elif "B-" in label.upper():
                    start_b = True
            if start_b:
                if start_s:
                    self.tagScheme = "BMES"
                else:
                    self.tagScheme = "BIO"

    def build_gaz_file(self, gaz_file):
        # build gaz file, initial read gaz embedding file
        if gaz_file:
            with codecs.open(gaz_file, 'r', 'utf-8') as fr:
                fins = fr.readlines()
                for fin in fins:
                    fin = fin.strip().split()[0]
                    if fin:
                        self.gaz.insert(fin, "one_source")
                print("Load gaz file: ", gaz_file, " total size:", self.gaz.size())
        else:
            print('[' + sys._getframe().f_code.co_name + '] ' + "Gaz file is None, load nothing")

    def build_gaz_alphabet(self, input_file):
        if input_file is None or not os.path.isfile(input_file):
            # print('[' + sys._getframe().f_code.co_name + '] file ' + str(input_file) + "can not be found or is not a file address")
            return
        with codecs.open(input_file, 'r', 'utf-8') as fr:
            in_lines = fr.readlines()
            char_list = []
            for line in in_lines:
                if len(line) > 3:
                    char = line.split()[0]
                    if self.number_normalized:
                        char = normalize_char(char)
                    char_list.append(char)
                else:
                    c_length = len(char_list)
                    for idx in range(c_length):
                        matched_entity = self.gaz.enumerateMatchList(char_list[idx:])
                        for entity in matched_entity:
                            # print entity, self.gaz.searchId(entity),self.gaz.searchType(entity)
                            self.gaz_alphabet.add(entity)
                    char_list = []
            print("gaz alphabet size:", self.gaz_alphabet.size())

    # Alphabet
    def fix_alphabet(self):
        self.char_alphabet.close()  # alphabet.keep_growing=False
        self.bichar_alphabet.close()
        self.label_alphabet.close()
        self.gaz_alphabet.close()

    def build_char_pretrain_emb(self, emb_path):
        print("build word pretrain emb...")
        self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(emb_path, self.char_alphabet, self.char_emb_dim, self.norm_char_emb)

    def build_bichar_pretrain_emb(self, emb_path):
        print("build bichar pretrain emb...")
        self.pretrain_bichar_embedding, self.bichar_emb_dim = build_pretrain_embedding(emb_path, self.bichar_alphabet, self.bichar_emb_dim,
                                                                                       self.norm_bichar_emb)

    def build_gaz_pretrain_emb(self, emb_path):
        print("build gaz pretrain emb...")
        self.pretrain_gaz_embedding, self.gaz_emb_dim = build_pretrain_embedding(emb_path, self.gaz_alphabet, self.gaz_emb_dim, self.norm_gaz_emb)

    def generate_instance(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_seg_instance(input_file, self.char_alphabet, self.bichar_alphabet,
                                                                 self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_seg_instance(input_file, self.char_alphabet, self.bichar_alphabet,
                                                             self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_seg_instance(input_file, self.char_alphabet, self.bichar_alphabet,
                                                               self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_seg_instance(input_file, self.char_alphabet, self.bichar_alphabet,
                                                             self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % name)

    def generate_instance_with_gaz(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance_with_gaz(input_file, self.gaz, self.char_alphabet, self.bichar_alphabet,
                                                                      self.gaz_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance_with_gaz(input_file, self.gaz, self.char_alphabet, self.bichar_alphabet,
                                                                  self.gaz_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance_with_gaz(input_file, self.gaz, self.char_alphabet, self.bichar_alphabet,
                                                                    self.gaz_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance_with_gaz(input_file, self.gaz, self.char_alphabet, self.bichar_alphabet,
                                                                  self.gaz_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % name)

    def generate_instance_with_gaz_2(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance_with_gaz_2(self.HP_num_layer, input_file, self.gaz, self.char_alphabet, self.bichar_alphabet,
                                                                       self.gaz_alphabet, self.label_alphabet, self.number_normalized,
                                                                      self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance_with_gaz_2(self.HP_num_layer, input_file, self.gaz, self.char_alphabet, self.bichar_alphabet,
                                                                  self.gaz_alphabet, self.label_alphabet, self.number_normalized,
                                                                  self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance_with_gaz_2(self.HP_num_layer, input_file, self.gaz, self.char_alphabet, self.bichar_alphabet,
                                                                     self.gaz_alphabet, self.label_alphabet, self.number_normalized,
                                                                    self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance_with_gaz_2(self.HP_num_layer, input_file, self.gaz, self.char_alphabet, self.bichar_alphabet,
                                                                   self.gaz_alphabet, self.label_alphabet, self.number_normalized,
                                                                  self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % (name))


    def write_decoded_results(self, output_file, predict_results, name):
        fout = open(output_file, 'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert (sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                ## content_list[idx] is a list with [word, char, label]
                fout.write(content_list[idx][0][idy].encode('utf-8') + " " + predict_results[idx][idy] + '\n')

            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s" % (name, output_file))
