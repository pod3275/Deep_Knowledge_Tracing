# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:50:02 2021

@author: LSH
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.tools import freeze_graph


def generate_freezed_graph():
    freeze_graph.freeze_graph('./results/model/model.pb',"", True, './results/model/LSTM-102',
                              'output_layer/preds', "save/restore_all", "save/Const", 
                              './results/model/frozen_model.pb', True, "")

if __name__ == "__main__":
    generate_freezed_graph()