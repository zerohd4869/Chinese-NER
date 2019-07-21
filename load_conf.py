import json
import os
import sys
import argparse
import configparser


def load_conf(conf_filename):
    param_conf_dict = {}
    cf = configparser.ConfigParser()
    cf.read(conf_filename)
    int_conf_keys = {'hyperparams': ["HP_batch_size",
                                     "HP_iteration",
                                     "HP_char_hidden_dim",
                                     "HP_hidden_dim",
                                     "HP_lstm_layer"],
                     'model_params': ["char_emb_dim",
                                      "bichar_emb_dim",
                                      "gaz_emb_dim",
                                      "label_size",
                                      "char_alphabet_size",
                                      "bichar_alphabet_size",
                                      "character_alphabet_size",
                                      "label_alphabet_size",
                                      "MAX_SENTENCE_LENGTH",
                                      "MAX_WORD_LENGTH"
                                      ]}

    for session_key in int_conf_keys:
        for option_key in int_conf_keys[session_key]:
            try:
                option_value = cf.get(session_key, option_key)
                param_conf_dict[option_key] = int(option_value)
            except:
                raise ValueError("%s--%s is not a integer" % (session_key, option_key))

    float_conf_keys = {'hyperparams': ["HP_dropout",
                                       "HP_lr",
                                       "HP_lr_decay",
                                       "HP_clip",
                                       "HP_momentum"
                                       ],
                       "model_params": ["gaz_dropout"
                                        ]}
    for session_key in float_conf_keys:
        for option_key in float_conf_keys[session_key]:
            try:
                option_value = cf.get(session_key, option_key)
                param_conf_dict[option_key] = float(option_value)
            except:
                raise ValueError("%s--%s is not a float" % (session_key, option_key))

    bool_conf_keys = {'model_params': ["seg",
                                       "use_bichar",
                                       "number_normalized",
                                       "norm_char_emb",
                                       "norm_bichar_emb",
                                       "norm_gaz_emb",
                                       "gaz_lower"],
                      'hyperparams': ["HP_bilstm",
                                      "HP_gpu",
                                      "HP_fix_gaz_emb",
                                      "HP_use_gaz"]}
    for session_key in bool_conf_keys:
        for option_key in bool_conf_keys[session_key]:
            try:
                option_value = cf.get(session_key, option_key)
                temp = str(option_value)
                if temp == 'True':
                    param_conf_dict[option_key] = True
                else:
                    param_conf_dict[option_key] = False
            except:
                raise ValueError("%s--%s is not a bool" % (session_key, option_key))

    str_conf_keys = {
        'save_dir': ['status',
                     "save_model",
                     "save_dset",
                     "load_model",
                     "output",
                     "model_path"],
        'dataset': ["train",
                    "dev",
                    "test"],
        'model_params': ["model",
                         "embedding",
                         "char_emb",
                         "bichar_emb",
                         "gaz_file",
                         "pretrain_char_embedding",
                         "pretrain_bichar_embedding",
                         "pretrain_gaz_embedding"
                         ]
    }

    for session_key in str_conf_keys:
        for option_key in str_conf_keys[session_key]:
            try:
                param_conf_dict[option_key] = cf.get(session_key, option_key)
            except:
                raise ValueError("%s no such option %s" % (session_key, option_key))

    if not os.path.exists(param_conf_dict['model_path']):
        os.mkdir(param_conf_dict['model_path'])

    return param_conf_dict
