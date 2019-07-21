import sys
import os
import codecs
import json
from tqdm import tqdm
import re
import time

print("py_version:", sys.version, "\npath:", os.getcwd())


# CoNLL format (prefer BIOES tag scheme), with each character its label for one line.
# Sentences are splited with a null line.
def json2conll(input_data_dir='./util_test/raw_data_0.json',
               output_data_dir="./util_test/processed_data_0.bmes",
               schemas_dir='./util_test/all_50_schemas',
               entity_type_dir='./util_test/all_entity_type.json',
               predicate_dir='./util_test/all_predicate.json'):
    start_time = time.time()
    labeling_list = []
    so_taging_list = []
    chars_list = []
    e2etype = {}
    entity_type = set()
    all_50_schemas = set()
    with codecs.open(input_data_dir, 'r', encoding='utf-8') as f:
        for l in f:
            # json.loads()：将str数据转化成dict数据
            # json.dumps()：将dict数据转化成str数据
            # json.dump/load() 读写json文件函数
            a = json.loads(l)
            chars = [char.strip() for char in a['text']]
            for i in a['spo_list']:
                for j in ['subject', 'object']:
                    entity = re.sub(u'[^\u4e00-\u9fa5a-zA-Z0-9]', u'', i[j].upper())
                    if not entity in e2etype:
                        e2etype[entity] = i[j + '_type']
    with open(schemas_dir) as f:
        for l in f:
            a = json.loads(l)
            all_50_schemas.add(a['predicate'])
            for j in ['subject', 'object']:
                entity_type.add(a[j + '_type'])
    id2predicate = {i + 1: j for i, j in enumerate(all_50_schemas)}  # 0表示终止类别
    predicate2id = {j: i for i, j in id2predicate.items()}
    id2etype = {i + 1: j for i, j in enumerate(entity_type)}  # 0表示终止类别
    etype2id = {j: i for i, j in id2etype.items()}
    e2etypeid = {i: etype2id[j] for i, j in e2etype.items()}
    with codecs.open(predicate_dir, 'w', encoding='utf-8') as f:
        json.dump([id2predicate, predicate2id], f, indent=4, ensure_ascii=False)
    with codecs.open(entity_type_dir, 'w', encoding='utf-8') as f:
        json.dump([id2etype, etype2id], f, indent=4, ensure_ascii=False)
    with codecs.open(input_data_dir, 'r', encoding='utf-8') as f:
        for l in tqdm(f):
            a = json.loads(l)
            a['text'] = re.sub(u'[^\u4e00-\u9fa5a-zA-Z0-9]', u'', a['text'].upper())
            chars = [char.strip() for char in a['text']]
            labels = [''] * len(chars)
            so_tags = [''] * len(chars)
            entity = ''
            if 'spo_list' in a:
                for i in a['spo_list']:
                    for j in ['subject', 'object']:
                        try:
                            entity = re.sub(u'[^\u4e00-\u9fa5a-zA-Z0-9]', u'', i[j].upper())
                            idx = a['text'].index(entity.split()[0])
                            etypeid = str(e2etypeid[entity])
                            predicate_id = str(predicate2id[i['predicate']])
                            if 1 <= len(entity):

                                if 1 == len(entity):
                                    labels[idx] += ',S-' + etypeid + ' '
                                    if j == "subject":
                                        so_tags[idx] += ',' + predicate_id + '-e1,'
                                    else:
                                        so_tags[idx] += ',' + predicate_id + '-e2,'
                                elif 1 < len(entity):
                                    for k, l in enumerate(entity):
                                        m_idx = idx + k
                                        if k == 0:
                                            labels[m_idx] += ',B-' + etypeid
                                        elif k == (len(i[j]) - 1):
                                            labels[m_idx] += ',E-' + etypeid
                                        else:
                                            labels[m_idx] += ',M-' + etypeid
                                        if j == 'subject':
                                            so_tags[m_idx] += ',' + predicate_id + '-e1'
                                        else:
                                            so_tags[m_idx] += ',' + predicate_id + '-e2'
                        except:
                            continue
            labeling_list.append(labels)
            so_taging_list.append(so_tags)
            chars_list.append(chars)
    with open(output_data_dir, 'w') as fp:
        for idx in range(len(labeling_list)):
            for idy in range(len(labeling_list[idx])):
                if labeling_list[idx][idy] == '':
                    labeling_list[idx][idy] = 'O'
                else:
                    labeling_list[idx][idy] = labeling_list[idx][idy][1:]
                if so_taging_list[idx][idy] == '':
                    so_taging_list[idx][idy] = 'null'
                else:
                    so_taging_list[idx][idy] = so_taging_list[idx][idy][1:]
                fp.write(chars_list[idx][idy] + " " + labeling_list[idx][idy] + " " + so_taging_list[idx][idy] + '\n')
            fp.write('\n')
    time_cost = time.time() - start_time
    print("json2conll_time_cost: %2fs" % time_cost)


if __name__ == '__main__':
    json2conll('./util_test/dev_data.json')
