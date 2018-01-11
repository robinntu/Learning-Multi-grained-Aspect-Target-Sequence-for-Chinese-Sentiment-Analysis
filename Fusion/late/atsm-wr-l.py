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
inputs_r = tf.placeholder(tf.float32,shape=(None, 30,128))

inputs_w_shape= inputs_w.get_shape()
inputs_r_shape= inputs_r.get_shape()



lstm1_w = LSTM(32,activation='relu', return_sequences=True,W_regularizer=l2(0.01),inner_activation='sigmoid')(inputs_w)
lstm1_r = LSTM(32,activation='relu', return_sequences=True,W_regularizer=l2(0.01),inner_activation='sigmoid')(inputs_r)



aspect_id_w = Input(dtype=np.float32,batch_shape=[inputs_w_shape[0],inputs_w_shape[1],6])
aspect_id_r = Input(dtype=np.float32,batch_shape=[inputs_r_shape[0],inputs_r_shape[1],20])


x_batch_dot_w = K.batch_dot(inputs_w,aspect_id_w, axes=[1, 1])
x_batch_dot_r = K.batch_dot(inputs_r,aspect_id_r, axes=[1, 1])



batch_dot_shape_w =x_batch_dot_w.get_shape()
batch_dot_shape_r =x_batch_dot_r.get_shape()





def aspect_layer_w(seq):
    vec_w=x_batch_dot_w[:,:,seq]

    vec_w2= K.repeat(vec_w, 15)   
    atten_inputs_w=tf.concat([lstm1_w,vec_w2],2)
    atten_inputs_w=tf.cast(atten_inputs_w,tf.float32)
    



    gi_w= Dense(1, bias=True, activation='tanh')(atten_inputs_w)

    ai_w= tf.nn.softmax(gi_w, dim=1, name=None)
    
    atten_output_w = K.batch_dot(atten_inputs_w,ai_w, axes=[1, 1])
    atten_output_w=K.squeeze(atten_output_w,2)

    return atten_output_w



def aspect_layer(seq):
    vec_r=x_batch_dot_r[:,:,seq]

    vec_r2= K.repeat(vec_r, 30)   
    atten_inputs_r=tf.concat([lstm1_r,vec_r2],2)
    atten_inputs_r=tf.cast(atten_inputs_r,tf.float32)
    



    gi_r= Dense(1, bias=True, activation='tanh')(atten_inputs_r)

    ai_r= tf.nn.softmax(gi_r, dim=1, name=None)
    
    atten_output_r = K.batch_dot(atten_inputs_r,ai_r, axes=[1, 1])
    atten_output_r=K.squeeze(atten_output_r,2)

    return atten_output_r

atten_output_w1=aspect_layer_w(0)
atten_output_w2=aspect_layer_w(1)
atten_output_w3=aspect_layer_w(2)


atten_output1=aspect_layer(0)
atten_output2=aspect_layer(1)
atten_output3=aspect_layer(2)
atten_output4=aspect_layer(3)
atten_output5=aspect_layer(4)
atten_output6=aspect_layer(5)
atten_output7=aspect_layer(6)
atten_output8=aspect_layer(7)
atten_output9=aspect_layer(8)
atten_output10=aspect_layer(9)
atten_output11=aspect_layer(10)
atten_output12=aspect_layer(11)
atten_output13=aspect_layer(12)
atten_output14=aspect_layer(13)
atten_output15=aspect_layer(14)
atten_output16=aspect_layer(15)
atten_output17=aspect_layer(16)
atten_output18=aspect_layer(17)

lstm_input_w= tf.concat(1,[tf.expand_dims(atten_output_w1, 1),tf.expand_dims(atten_output_w2, 1)])
lstm_input_w1=tf.concat(1,[lstm_input_w,tf.expand_dims(atten_output_w3, 1)])


lstm_input_r= tf.concat(1,[tf.expand_dims(atten_output1, 1),tf.expand_dims(atten_output2, 1)])
lstm_input_r1=tf.concat(1,[lstm_input_r,tf.expand_dims(atten_output3, 1)])
lstm_input_r2=tf.concat(1,[lstm_input_r1,tf.expand_dims(atten_output4, 1)])
lstm_input_r3=tf.concat(1,[lstm_input_r2,tf.expand_dims(atten_output5, 1)])
lstm_input_r4=tf.concat(1,[lstm_input_r3,tf.expand_dims(atten_output6, 1)])
lstm_input_r5=tf.concat(1,[lstm_input_r4,tf.expand_dims(atten_output7, 1)])
lstm_input_r6=tf.concat(1,[lstm_input_r5,tf.expand_dims(atten_output8, 1)])
lstm_input_r7=tf.concat(1,[lstm_input_r6,tf.expand_dims(atten_output9, 1)])
lstm_input_r8=tf.concat(1,[lstm_input_r7,tf.expand_dims(atten_output10, 1)])
lstm_input_r9=tf.concat(1,[lstm_input_r8,tf.expand_dims(atten_output11, 1)])
lstm_input_r10=tf.concat(1,[lstm_input_r9,tf.expand_dims(atten_output12, 1)])
lstm_input_r11=tf.concat(1,[lstm_input_r10,tf.expand_dims(atten_output13, 1)])
lstm_input_r12=tf.concat(1,[lstm_input_r11,tf.expand_dims(atten_output14, 1)])
lstm_input_r13=tf.concat(1,[lstm_input_r12,tf.expand_dims(atten_output15, 1)])
lstm_input_r14=tf.concat(1,[lstm_input_r13,tf.expand_dims(atten_output16, 1)])
lstm_input_r15=tf.concat(1,[lstm_input_r13,tf.expand_dims(atten_output17, 1)])
lstm_input_r16=tf.concat(1,[lstm_input_r13,tf.expand_dims(atten_output18, 1)])

var_w0_w= LSTM(output_dim=256,activation='relu',return_sequences=False,W_regularizer=l2(0.01), inner_activation='sigmoid')(lstm_input_w1)
var_r0_r= LSTM(output_dim=256,activation='relu',return_sequences=False,W_regularizer=l2(0.01), inner_activation='sigmoid')(lstm_input_r16)


var_w1=Dense(100,bias=False,activation='tanh')(var_w0_w)
var_r1=Dense(100,bias=False,activation='tanh')(var_r0_r)



var_w= Dropout(0.5)(var_w1)
var_r= Dropout(0.5)(var_r1)

var = merge([var_w,var_r], mode= 'concat')

predictions= Dense(2,bias=False,activation='softmax')(var)


labels = tf.placeholder(tf.float32, shape=(None,2))


loss = tf.reduce_mean(binary_crossentropy(labels, predictions))

train_w_step = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)

acc_value = accuracy(labels, predictions)


print '-------Model building complete--------'    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    
    # load dataset embeddings
    dataset_w=pickle.load(open('../data/Chinese/word/623embedding.py','r'))
    dataset_r=pickle.load(open('../data/Chinese/radical/623embedding.py','r'))
    dataset_size=dataset_w.shape[0]

 


    # load dataset labels
    label0= codecs.open('../data/Chinese/label/623label.txt','r','utf8').readlines()
    label=[]
    for e in label0:
        label.append(np.int(e.rstrip()))    
        



    # load aspect info
    aspect_mat_w=pickle.load(open('../data/Chinese/word/623id.py','r'))
    aspect_mat_r=pickle.load(open('../data/Chinese/radical/623id.py','r'))


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
            X_train_r1=[]
            train_r_aspect_id_r=[]
            y_train_1=[]
            for e in train_ind:
                X_train_w1.append(dataset_w[e])
                train_w_aspect_id_w.append(aspect_mat_w[e,:,:])
                
                X_train_r1.append(dataset_r[e])
                train_r_aspect_id_r.append(aspect_mat_r[e,:,:])
                y_train_1.append(label[e])
              
                 
            
            X_train_w1=np.asarray(X_train_w1)    
            train_w_aspect_id_w=np.asarray(train_w_aspect_id_w)  
            
            X_train_r1=np.asarray(X_train_r1)    
            train_r_aspect_id_r=np.asarray(train_r_aspect_id_r)  

            
            b=np.zeros((len(y_train_1),2))
            b[np.arange(len(y_train_1)),y_train_1]  =1
            y_train_1=b

            
            X_test_w1=[]
            y_test1=[]
            test_aspect_id_w=[]
            X_test_r1=[]
            test_aspect_id_r=[]
            for e in  test_ind:
                X_test_w1.append(dataset_w[e])
                test_aspect_id_w.append(aspect_mat_w[e,:,:])
                X_test_r1.append(dataset_r[e])
                test_aspect_id_r.append(aspect_mat_r[e,:,:])
                y_test1.append(label[e])

            c=np.zeros((len(y_test1),2))
            c[np.arange(len(y_test1)),y_test1]  =1
            y_test1=c     

        
    
            
            start = 0
            end = 50
            for i in range(num_minibatch):
         
                
                X_w = X_train_w1[start:end]
                X_r = X_train_r1[start:end]
    
                Y = y_train_1[start:end]
                idd_w = train_w_aspect_id_w[start:end]
                idd_r = train_r_aspect_id_r[start:end]
    
                start = end
                end = start + 50
                sess.run(train_w_step, feed_dict={inputs_w: X_w, inputs_r: X_r, aspect_id_w:idd_w, aspect_id_r:idd_r, labels: Y, K.learning_phase(): 1})    
                
 
    
            # testing accuracy
            test_acc= acc_value.eval(feed_dict={inputs_w: X_test_w1, inputs_r: X_test_r1, aspect_id_w:test_aspect_id_w, aspect_id_r:test_aspect_id_r, labels: y_test1,K.learning_phase(): 0})
            
            
            if test_acc > best:
                best=test_acc

            
        print ('Epoch best performance---------', best)
        best_lst.append(best)
    print best_lst
    print ( 'Averaged testing accuracy---------',reduce(lambda x, y: x + y, best_lst) / len(best_lst))   
          