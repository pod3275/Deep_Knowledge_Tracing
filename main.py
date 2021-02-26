# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:16:30 2020

@author: LSH
"""

import os
import tensorflow as tf
import time

from utils import DKT
from load_data import DKTData

rnn_cells = {
    "LSTM": tf.contrib.rnn.LSTMCell,
    "GRU": tf.contrib.rnn.GRUCell,
    "BasicRNN": tf.contrib.rnn.BasicRNNCell,
    "LayerNormBasicLSTM": tf.contrib.rnn.LayerNormBasicLSTMCell,
}

num_runs = 1
num_epochs = 25
batch_size = 32
keep_prob = 0.8656542586183774

network_config = {}
network_config['batch_size'] = batch_size
network_config['hidden_layer_structure'] = [102, ]
network_config['learning_rate'] = 0.004155923499457689
network_config['keep_prob'] = keep_prob
network_config['rnn_cell'] = rnn_cells['LSTM']
network_config['max_grad_norm'] = 5.0
network_config['lambda_w1'] = 0.03
network_config['lambda_w2'] = 3.0
network_config['lambda_o'] = 0.1

train_path = './data/i-scream/i-scream_train.csv'
valid_path = './data/i-scream/i-scream_valid.csv'
test_path = './data/i-scream/i-scream_test.csv'
save_dir_prefix = './results/'

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    data = DKTData(train_path, valid_path, test_path, batch_size=batch_size)
    data_train = data.train
    data_valid = data.valid
    data_test = data.test
    num_problems = data.num_problems

    dkt = DKT(sess, data_train, data_valid, data_test, num_problems, network_config,
              save_dir_prefix=save_dir_prefix,
              num_runs=num_runs, num_epochs=num_epochs,
              keep_prob=keep_prob, logging=True, save=True)

    # run optimization of the created model
    dkt.model.build_graph()
    dkt.run_optimization()
    
    # close the session
    sess.close()

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("program run for: {0}s".format(end_time - start_time))
