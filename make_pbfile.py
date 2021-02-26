# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:50:02 2021

@author: LSH
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def one_hot(indices, depth=1865):
    encoding = tf.concat((tf.eye(depth), [tf.zeros(depth)]), axis=0)
    zero_condition = tf.equal(indices, 0)
    new_indices = tf.where(zero_condition, tf.fill(tf.shape(indices), depth+1), indices)
    index_condition = tf.not_equal(new_indices, -1)
    new_indices = tf.where(index_condition, tf.math.add(new_indices, -1), new_indices)
    return tf.gather_nd(encoding, new_indices)


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph


def makeSavedModel(model_name='dkt'):
    sess = tf.Session()
    sess.as_default()
    
    graph_model = load_graph('./results/model/frozen_model.pb')
    graph_model_def = graph_model.as_graph_def()
    
    new_graph = tf.Graph()
    new_graph.as_default()
    
    service_input = tf.placeholder(tf.int32, [None, None, 2], name='Input_index')
    prob_input = tf.expand_dims(service_input[:,:,0],2)
    model_input_prob = one_hot(prob_input)
    
    corr_input = prob_input * tf.expand_dims(service_input[:,:,1],2)
    model_input_corr = one_hot(corr_input)
    
    model_input = tf.concat((model_input_prob, model_input_corr), axis=2)
    tf.import_graph_def(graph_model_def, name='', input_map={"X":model_input})
    
    # set input/output of .savedmodel file
    preds = sess.graph.get_tensor_by_name('output_layer/preds:0')
    
    inputs = {'input':service_input}
    outputs = {'preds':preds[:,-1,:]}
    
    save_location = 'savedmodel'
    builder = tf.saved_model.builder.SavedModelBuilder(save_location)
    
    signature = tf.saved_model.signature_def_utils.predict_signature_def(                                                                        
        inputs=inputs, outputs=outputs)
    
    builder.add_meta_graph_and_variables(                                                                                                        
        sess=sess,                                                                                                                    
        tags=[tf.saved_model.tag_constants.SERVING],                                                                                             
        signature_def_map={                                                                                                                      
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:                                                                
                signature                                                                                                                        
        })
                                                                                                                                     
    builder.save()
    print("pb file created.")
    
    
if __name__ == "__main__":
    makeSavedModel('dkt')
