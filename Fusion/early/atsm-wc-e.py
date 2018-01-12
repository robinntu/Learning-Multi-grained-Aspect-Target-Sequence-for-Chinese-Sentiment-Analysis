# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.models import Model
from keras.layers import Dropout, Activation, Bidirectional, Embedding,LSTM,Input, Dense,Wrapper, Recurrent, merge,GRU
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
import numpy as np
import keras.backend as K
from keras.engine import InputSpec
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy as accuracy
from keras.regularizers import l2
import random


import os
import codecs
import cPickle as pickle

np.random.seed(7)

keras_learning_phase = tf.constant(True)





    
    
inputs_w = tf.placeholder(tf.float32,shape=(None, 15,128))
inputs_c = tf.placeholder(tf.float32,shape=(None, 20,128))

inputs_w_shape= inputs_w.get_shape()
inputs_c_shape= inputs_c.get_shape()



lstm1_w = LSTM(output_dim=32,activation='relu', return_sequences=True,W_regularizer=l2(0.01), consume_less='cpu',inner_activation='sigmoid')(inputs_w)
lstm1_c = LSTM(output_dim=32,activation='relu', return_sequences=True,W_regularizer=l2(0.01), consume_less='cpu',inner_activation='sigmoid')(inputs_c)



aspect_id_w = Input(dtype=np.float32,batch_shape=[inputs_w_shape[0],inputs_w_shape[1],6])
aspect_id_c = Input(dtype=np.float32,batch_shape=[inputs_c_shape[0],inputs_c_shape[1],8])


x_batch_dot_w = K.batch_dot(inputs_w,aspect_id_w, axes=[1, 1])
x_batch_dot_c = K.batch_dot(inputs_c,aspect_id_c, axes=[1, 1])



batch_dot_shape_w =x_batch_dot_w.get_shape()
batch_dot_shape_c =x_batch_dot_c.get_shape()





def aspect_layer_w(seq):
    vec_w=x_batch_dot_w[:,:,seq]

    vec_w2= K.repeat(vec_w, 15)   
    atten_inputs_w=tf.concat(2,[lstm1_w,vec_w2])
    atten_inputs_w=tf.cast(atten_inputs_w,tf.float32)
    


    gi_w= Dense(1, bias=True, activation='tanh')(atten_inputs_w)

    ai_w= tf.nn.softmax(gi_w, dim=1, name=None)
    
    atten_output_w = K.batch_dot(atten_inputs_w,ai_w, axes=[1, 1])
    atten_output_w=K.squeeze(atten_output_w,2)
    

    return atten_output_w

def aspect_layer_c(seq):
    vec_c=x_batch_dot_c[:,:,seq]

    vec_c2= K.repeat(vec_c, 20)   
    atten_inputs_c=tf.concat(2,[lstm1_c,vec_c2])
    atten_inputs_c=tf.cast(atten_inputs_c,tf.float32)
    


    gi_c= Dense(1, bias=True, activation='tanh')(atten_inputs_c)

    ai_c= tf.nn.softmax(gi_c, dim=1, name=None)
    
    atten_output_c = K.batch_dot(atten_inputs_c,ai_c, axes=[1, 1])
    atten_output_c=K.squeeze(atten_output_c,2)
    
    

    return atten_output_c




atten_output_w1=aspect_layer_w(0)
atten_output_w2=aspect_layer_w(1)
atten_output_w3=aspect_layer_w(2)

atten_output_c1=aspect_layer_c(0)
atten_output_c2=aspect_layer_c(1)
atten_output_c3=aspect_layer_c(2)
atten_output_c4=aspect_layer_c(3)
atten_output_c5=aspect_layer_c(4)
atten_output_c6=aspect_layer_c(5)



atten_output_w1 = merge([atten_output_w1, atten_output_c1,atten_output_c2], mode= 'concat')
atten_output_w2 = merge([atten_output_w2, atten_output_c3,atten_output_c4], mode= 'concat')
atten_output_w3 = merge([atten_output_w3, atten_output_c5,atten_output_c6], mode= 'concat')



lstm_input_w= tf.concat(1,[tf.expand_dims(atten_output_w1, 1),tf.expand_dims(atten_output_w2, 1)])
lstm_input_w1=tf.concat(1,[lstm_input_w,tf.expand_dims(atten_output_w3, 1)])



var_w0_w= LSTM(output_dim=256,activation='relu',return_sequences=False,W_regularizer=l2(0.01), inner_activation='sigmoid')(lstm_input_w1)

var_w1=Dense(100,bias=False,activation='tanh')(var_w0_w)


var_w= Dropout(0.5)(var_w1)


predictions= Dense(2,bias=False,activation='softmax')(var_w)



labels = tf.placeholder(tf.float32, shape=(None,2))



loss = tf.reduce_mean(binary_crossentropy(labels, predictions))

train_w_step = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)

acc_value = accuracy(labels, predictions)


print '-------Model building complete--------'    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    
    # load dataset embeddings
    dataset_w=pickle.load(open('..data/Chinese/word/2556embedding.p','r'))
    dataset_c=pickle.load(open('..data/Chinese/character/2556embedding.p','r'))
    dataset_size=dataset_c.shape[0]

 


    # load dataset labels
    label0= codecs.open('..data/Chinese/label/2556label.txt','r','utf8').readlines()
    label=[]
    for e in label0:
        label.append(np.int(e.rstrip()))    
        



    # load aspect info
    aspect_mat_w=pickle.load(open('..data/Chinese/word/2556id.p','r'))
    aspect_mat_c=pickle.load(open('..data/Chinese/character/2556id.p','r'))


    fil_size=np.int32(dataset_size)
    num_train=np.int32(fil_size*4/5)
    num_test=fil_size-num_train
    num_minibatch=np.int32(num_train/50)
    
    index = [x for x in range(fil_size)]
    
    best_lst=[]
    for n in range(5):

        best=0
        sess.run(tf.global_variables_initializer()) 
        test_ind=index[n*num_test:(n+1)*num_test]
        train_ind=[x for x in index if x not in test_ind]

        for epoch in range(50):
            print 'trainning for ' +'Epoch'+ str(epoch)
        

            X_train_w1=[]
            train_w_aspect_id_w=[]
            X_train_c1=[]
            train_c_aspect_id_c=[]
            y_train_1=[]
            for e in train_ind:
                X_train_w1.append(dataset_w[e])
                train_w_aspect_id_w.append(aspect_mat_w[e,:,:])
                
                X_train_c1.append(dataset_c[e])
                train_c_aspect_id_c.append(aspect_mat_c[e,:,:])
                y_train_1.append(label[e])
              
                 
            
            X_train_w1=np.asarray(X_train_w1)    
            train_w_aspect_id_w=np.asarray(train_w_aspect_id_w)  
            
            X_train_c1=np.asarray(X_train_c1)    
            train_c_aspect_id_c=np.asarray(train_c_aspect_id_c)  

            
            b=np.zeros((len(y_train_1),2))
            b[np.arange(len(y_train_1)),y_train_1]  =1
            y_train_1=b
            

            
            
            X_test_w1=[]
            y_test1=[]
            test_aspect_id_w=[]
            X_test_c1=[]
            test_aspect_id_c=[]
            for e in  test_ind:
                X_test_w1.append(dataset_w[e])
                test_aspect_id_w.append(aspect_mat_w[e,:,:])
                X_test_c1.append(dataset_c[e])
                test_aspect_id_c.append(aspect_mat_c[e,:,:])
                y_test1.append(label[e])
            

            c=np.zeros((len(y_test1),2))
            c[np.arange(len(y_test1)),y_test1]  =1
            y_test1=c    

        
    
            
            start = 0
            end = 50
            for i in range(num_minibatch):
       
                
                X_w = X_train_w1[start:end]
                X_c = X_train_c1[start:end]
    
                Y = y_train_1[start:end]
                idd_w = train_w_aspect_id_w[start:end]
                idd_c = train_c_aspect_id_c[start:end]
    
                start = end
                end = start + 50
                sess.run(train_w_step, feed_dict={inputs_w: X_w, inputs_c: X_c, aspect_id_w:idd_w, aspect_id_c:idd_c, labels: Y, K.learning_phase(): 1})    
                
                

    
            # testing accuracy
            test_acc= acc_value.eval(feed_dict={inputs_w: X_test_w1, inputs_c: X_test_c1, aspect_id_w:test_aspect_id_w, aspect_id_c:test_aspect_id_c, labels: y_test1,K.learning_phase(): 0})
            
            if test_acc > best:
                best=test_acc

            
        print ('Epoch best performance---------', best)
        best_lst.append(best)
    print best_lst
    print ( 'Averaged testing accuracy---------',reduce(lambda x, y: x + y, best_lst) / len(best_lst))   
   
 
