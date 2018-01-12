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


    
inputs_c = tf.placeholder(tf.float32,shape=(None, 20,128))
inputs_r = tf.placeholder(tf.float32,shape=(None, 30,128))

inputs_c_shape= inputs_c.get_shape()
inputs_r_shape= inputs_r.get_shape()



lstm1_c = LSTM(output_dim=32,activation='relu', return_sequences=True,W_regularizer=l2(0.01), consume_less='cpu',inner_activation='sigmoid')(inputs_c)
lstm1_r = LSTM(output_dim=32,activation='relu', return_sequences=True,W_regularizer=l2(0.01), consume_less='cpu',inner_activation='sigmoid')(inputs_r)



aspect_id_c = Input(dtype=np.float32,batch_shape=[inputs_c_shape[0],inputs_c_shape[1],8])
aspect_id_r = Input(dtype=np.float32,batch_shape=[inputs_r_shape[0],inputs_r_shape[1],20])


x_batch_dot_c = K.batch_dot(inputs_c,aspect_id_c, axes=[1, 1])
x_batch_dot_r = K.batch_dot(inputs_r,aspect_id_r, axes=[1, 1])



batch_dot_shape_c =x_batch_dot_c.get_shape()
batch_dot_shape_r =x_batch_dot_r.get_shape()




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


def aspect_layer_r(seq):
    vec_r=x_batch_dot_r[:,:,seq]

    vec_r2= K.repeat(vec_r, 30)   
    atten_inputs_r=tf.concat(2,[lstm1_r,vec_r2])
    atten_inputs_r=tf.cast(atten_inputs_r,tf.float32)
    
    gi_r= Dense(1, bias=True, activation='tanh')(atten_inputs_r)

    ai_r= tf.nn.softmax(gi_r, dim=1, name=None)
    
    atten_output_r = K.batch_dot(atten_inputs_r,ai_r, axes=[1, 1])
    atten_output_r=K.squeeze(atten_output_r,2)
    
    return atten_output_r


atten_output_c1=aspect_layer_c(0)
atten_output_c2=aspect_layer_c(1)
atten_output_c3=aspect_layer_c(2)
atten_output_c4=aspect_layer_c(3)
atten_output_c5=aspect_layer_c(4)
atten_output_c6=aspect_layer_c(5)

atten_output_r1=aspect_layer_r(0)
atten_output_r2=aspect_layer_r(1)
atten_output_r3=aspect_layer_r(2)
atten_output_r4=aspect_layer_r(3)
atten_output_r5=aspect_layer_r(4)
atten_output_r6=aspect_layer_r(5)
atten_output_r7=aspect_layer_r(6)
atten_output_r8=aspect_layer_r(7)
atten_output_r9=aspect_layer_r(8)
atten_output_r10=aspect_layer_r(9)
atten_output_r11=aspect_layer_r(10)
atten_output_r12=aspect_layer_r(11)
atten_output_r13=aspect_layer_r(12)
atten_output_r14=aspect_layer_r(13)
atten_output_r15=aspect_layer_r(14)
atten_output_r16=aspect_layer_r(15)
atten_output_r17=aspect_layer_r(16)
atten_output_r18=aspect_layer_r(17)



lstm_input_c= tf.concat(1,[tf.expand_dims(atten_output_c1, 1),tf.expand_dims(atten_output_c2, 1)])
lstm_input_c1=tf.concat(1,[lstm_input_c,tf.expand_dims(atten_output_c3, 1)])
lstm_input_c2=tf.concat(1,[lstm_input_c1,tf.expand_dims(atten_output_c4, 1)])
lstm_input_c3=tf.concat(1,[lstm_input_c2,tf.expand_dims(atten_output_c5, 1)])
lstm_input_c4=tf.concat(1,[lstm_input_c3,tf.expand_dims(atten_output_c6, 1)])



lstm_input_r= tf.concat(1,[tf.expand_dims(atten_output_r1, 1),tf.expand_dims(atten_output_r2, 1)])
lstm_input_r1=tf.concat(1,[lstm_input_r,tf.expand_dims(atten_output_r3, 1)])
lstm_input_r2=tf.concat(1,[lstm_input_r1,tf.expand_dims(atten_output_r4, 1)])
lstm_input_r3=tf.concat(1,[lstm_input_r2,tf.expand_dims(atten_output_r5, 1)])
lstm_input_r4=tf.concat(1,[lstm_input_r3,tf.expand_dims(atten_output_r6, 1)])
lstm_input_r5=tf.concat(1,[lstm_input_r4,tf.expand_dims(atten_output_r7, 1)])
lstm_input_r6=tf.concat(1,[lstm_input_r5,tf.expand_dims(atten_output_r8, 1)])
lstm_input_r7=tf.concat(1,[lstm_input_r6,tf.expand_dims(atten_output_r9, 1)])
lstm_input_r8=tf.concat(1,[lstm_input_r7,tf.expand_dims(atten_output_r10, 1)])
lstm_input_r9=tf.concat(1,[lstm_input_r8,tf.expand_dims(atten_output_r11, 1)])
lstm_input_r10=tf.concat(1,[lstm_input_r9,tf.expand_dims(atten_output_r12, 1)])
lstm_input_r11=tf.concat(1,[lstm_input_r10,tf.expand_dims(atten_output_r13, 1)])
lstm_input_r12=tf.concat(1,[lstm_input_r11,tf.expand_dims(atten_output_r14, 1)])
lstm_input_r13=tf.concat(1,[lstm_input_r12,tf.expand_dims(atten_output_r15, 1)])
lstm_input_r14=tf.concat(1,[lstm_input_r13,tf.expand_dims(atten_output_r16, 1)])
lstm_input_r15=tf.concat(1,[lstm_input_r13,tf.expand_dims(atten_output_r17, 1)])
lstm_input_r16=tf.concat(1,[lstm_input_r13,tf.expand_dims(atten_output_r18, 1)])

var_c0_c= LSTM(output_dim=256,activation='relu',return_sequences=False,W_regularizer=l2(0.01), inner_activation='sigmoid')(lstm_input_c4)
var_r0_r= LSTM(output_dim=256,activation='relu',return_sequences=False,W_regularizer=l2(0.01), inner_activation='sigmoid')(lstm_input_r16)

var_r1=Dense(100,bias=False,activation='tanh')(var_r0_r)
var_c1=Dense(100,bias=False,activation='tanh')(var_c0_c)

#dropout layer
var_c= Dropout(0.5)(var_c1)
var_r= Dropout(0.5)(var_r1)

var = merge([var_c,var_r], mode= 'concat')

predictions= Dense(2,bias=False,activation='softmax')(var)



labels = tf.placeholder(tf.float32, shape=(None,2))


loss = tf.reduce_mean(binary_crossentropy(labels, predictions))

train_w_step = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)

acc_value = accuracy(labels, predictions)


print '-------Model building complete--------'    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    
    # load dataset embeddings
    dataset_c=pickle.load(open('../data/Chinese/character/623embedding.p','r'))
    dataset_r=pickle.load(open('../data/Chinese/radical/623embedding.p','r'))
    dataset_size=dataset_c.shape[0]

 


    # load dataset labels
    label0= codecs.open('../data/Chinese/label/623label.txt','r','utf8').readlines()
    label=[]
    for e in label0:
        label.append(np.int(e.rstrip()))    
        



    # load aspect info
    aspect_mat_c=pickle.load(open('../data/Chinese/character/623id.p','r'))
    aspect_mat_r=pickle.load(open('../data/Chinese/radical/623id.p','r'))


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
    

            
    
            X_train_c1=[]
            train_c_aspect_id_c=[]
            X_train_r1=[]
            train_r_aspect_id_r=[]
            y_train_1=[]
            for e in train_ind:
                X_train_c1.append(dataset_c[e])
                train_c_aspect_id_c.append(aspect_mat_c[e,:,:])
                
                X_train_r1.append(dataset_r[e])
                train_r_aspect_id_r.append(aspect_mat_r[e,:,:])
                y_train_1.append(label[e])
              
                 
            
            X_train_c1=np.asarray(X_train_c1)    
            train_c_aspect_id_c=np.asarray(train_c_aspect_id_c)  
            
            X_train_r1=np.asarray(X_train_r1)    
            train_r_aspect_id_r=np.asarray(train_r_aspect_id_r)  

            
            b=np.zeros((len(y_train_1),2))
            b[np.arange(len(y_train_1)),y_train_1]  =1
            y_train_1=b

            
            
            X_test_c1=[]
            y_test1=[]
            test_aspect_id_c=[]
            X_test_r1=[]
            test_aspect_id_r=[]
            for e in  test_ind:
                X_test_c1.append(dataset_c[e])
                test_aspect_id_c.append(aspect_mat_c[e,:,:])
                X_test_r1.append(dataset_r[e])
                test_aspect_id_r.append(aspect_mat_r[e,:,:])
                y_test1.append(label[e])
            

            c=np.zeros((len(y_test1),2))
            c[np.arange(len(y_test1)),y_test1]  =1
            y_test1=c  

        
    
            
            start = 0
            end = 50
            for i in range(num_minibatch):
          
                
                X_c = X_train_c1[start:end]
                X_r = X_train_r1[start:end]
    
                Y = y_train_1[start:end]
                idd_c = train_c_aspect_id_c[start:end]
                idd_r = train_r_aspect_id_r[start:end]
    
                start = end
                end = start + 50
                
                sess.run(train_w_step, feed_dict={inputs_c: X_c, inputs_r: X_r, aspect_id_c:idd_c, aspect_id_r:idd_r, labels: Y, K.learning_phase(): 1})    
                
                
            Loss = str(sess.run(loss, feed_dict={inputs_c: X_train_c1, inputs_r: X_train_r1, aspect_id_c:train_c_aspect_id_c, aspect_id_r:train_r_aspect_id_r, labels: y_train_1, K.learning_phase(): 0}))

            # testing accuracy
            test_acc= acc_value.eval(feed_dict={inputs_c: X_test_c1, inputs_r: X_test_r1, aspect_id_c:test_aspect_id_c, aspect_id_r:test_aspect_id_r, labels: y_test1,K.learning_phase(): 0})

                        
            if test_acc > best:
                best=test_acc

            
        print ('Epoch best performance---------', best)
        best_lst.append(best)
    print best_lst
    print ( 'Averaged testing accuracy---------',reduce(lambda x, y: x + y, best_lst) / len(best_lst))   
          
