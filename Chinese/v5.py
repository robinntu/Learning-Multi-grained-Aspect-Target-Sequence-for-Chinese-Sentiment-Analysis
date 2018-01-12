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



import codecs
import cPickle as pickle

np.random.seed(7)

keras_learning_phase = tf.constant(True)

 
inputs = tf.placeholder(tf.float32,shape=(None, 15,128))

inputs_shape= inputs.get_shape()

lstm1 = LSTM(output_dim=32, return_sequences=True,W_regularizer=l2(0.01), consume_less='mem')(inputs)

aspect_id = Input(dtype=np.float32,batch_shape=[inputs_shape[0],inputs_shape[1],6])
x_batch_dot = K.batch_dot(inputs,aspect_id, axes=[1, 1])

batch_dot_shape =x_batch_dot.get_shape()

def aspect_layer(seq):
    vec=x_batch_dot[:,:,seq]
    vec2= K.repeat(vec, 15)   
    atten_inputs=tf.concat(2,[lstm1,vec2])
    atten_inputs=tf.cast(atten_inputs,tf.float32)
    
    gi= Dense(1, bias=True, activation='tanh')(atten_inputs)
    ai= tf.nn.softmax(gi, dim=1, name=None)
    atten_output = K.batch_dot(atten_inputs,ai, axes=[1, 1])
    atten_output=K.squeeze(atten_output,2)

    return atten_output





atten_output1=aspect_layer(0)
atten_output2=aspect_layer(1)
atten_output3=aspect_layer(2)




var0=merge([atten_output1,atten_output2,atten_output3], mode='concat')


var1=Dense(100,bias=False,activation='tanh')(var0)


var= Dropout(0.5)(var1)


predictions= Dense(2,bias=False,activation='softmax')(var)

labels = tf.placeholder(tf.float32, shape=(None,2))


loss = tf.reduce_mean(binary_crossentropy(labels, predictions))

train_step = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)

acc_value = accuracy(labels, predictions)


print '-------Model building complete--------'
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    # load dataset embeddings, change according to dataset
    dataset=pickle.load(open('../data/Chinese/word/1172embedding.p','r'))
    dataset_size=dataset.shape[0]
 


    # load dataset labels, change according to dataset
    label0= codecs.open('../data/Chinese/label/1172label.txt','r','utf8').readlines()
    label=[]
    for e in label0:
        label.append(np.int(e.rstrip()))    
        



    # load dataset aspect term id, change according to dataset
    aspect_mat=pickle.load(open('../data/Chinese/word/1172id.p','r'))

    
    
    fil_size=np.int32(dataset_size)
    num_train=np.int32(fil_size*4/5)
    num_test=fil_size-num_train
    num_minibatch=np.int32(num_train/50)
    
    index = [x for x in range(fil_size)]
    
    best_lst=[]
    for n in range(5):

        best=0
        sess.run(tf.global_variables_initializer())
        prf_mata=[0,0,0,0,0,0]
        prf_aver=[0,0,0,0,0,0]    
        test_ind=index[n*num_test:(n+1)*num_test]
        train_ind=[x for x in index if x not in test_ind]
        for epoch in range(50):
            print 'trainning for ' +'Epoch'+ str(epoch)

            X_train1=[]
            train_aspect_id=[]
            y_train1=[]
            for e in train_ind:
                X_train1.append(dataset[e])
                train_aspect_id.append(aspect_mat[e,:,:])
                y_train1.append(label[e])
              
                 
            
            X_train1=np.asarray(X_train1)    
            train_aspect_id=np.asarray(train_aspect_id)  

            
            b=np.zeros((len(y_train1),2))
            b[np.arange(len(y_train1)),y_train1]  =1
            y_train1=b
            
            
            
            X_test1=[]
            y_test1=[]
            test_aspect_id=[]
            noastest=[]
            for e in  test_ind:
                X_test1.append(dataset[e])
                test_aspect_id.append(aspect_mat[e,:,:])
                y_test1.append(label[e])
            
            

            c=np.zeros((len(y_test1),2))
            c[np.arange(len(y_test1)),y_test1]  =1
            y_test1=c    

        
    
            
            start = 0
            end = 50
            for i in range(num_minibatch):
           
                
                X = X_train1[start:end]
                Y = y_train1[start:end]
                idd = train_aspect_id[start:end]
                start = end
                end = start + 50
                sess.run(train_step, feed_dict={inputs: X,  aspect_id:idd, labels: Y, K.learning_phase(): 1})    
                
                

    
            # testing accuracy
            test_acc= acc_value.eval(feed_dict={inputs: X_test1,  aspect_id:test_aspect_id, labels: y_test1,K.learning_phase(): 0})
            pre=predictions.eval(feed_dict={inputs: X_test1,  aspect_id:test_aspect_id, labels: y_test1,K.learning_phase(): 0})
            

            
            predict=[]
            for entry in pre:
                if entry[0]>entry[1]:
                    predict.append(-1)
                else:
                    predict.append(1)
                    
            truth=[]
            for entry in y_test1:
                if entry[0]>entry[1]:
                    truth.append(-1)
                else:
                    truth.append(1)                    
            

            
            pp_denominator=0
            pp_numerator=0
            np_denominator=0
            np_numerator=0
            pr_denominator=0
            pr_numerator=0
            nr_denominator=0
            nr_numerator=0
            pos_pre=0
            pos_rec=0
            neg_pre=0
            neg_rec=0
            pos_f=0
            neg_f=0
            
            
            for t, p in zip(truth, predict):
                if p>0:
                    #pos precision
                    pp_denominator+=1
                    if t>0:
                        pp_numerator+=1
                    
                else:
                    #neg precision
                    np_denominator+=1
                    if t<0:
                        np_numerator+=1
                                          
                    
                if t>0:
                    #pos recall
                    pr_denominator+=1
                    if p>0:
                        pr_numerator+=1
                else:
                    #neg recall
                    nr_denominator+=1
                    if p<0:
                        nr_numerator+=1
                    
            if pp_denominator!=0:
                pos_pre=pp_numerator/float(pp_denominator)
            if pr_denominator !=0:
                pos_rec=pr_numerator/float(pr_denominator)
            if np_denominator !=0:
                neg_pre=np_numerator/float(np_denominator)
            if nr_denominator !=0:
                neg_rec=nr_numerator/float(nr_denominator)            
            
            if (pos_pre+pos_rec)!=0:
                pos_f= 2*pos_pre*pos_rec/(pos_pre+pos_rec)
            if (neg_pre+neg_rec)!=0:
                neg_f= 2*neg_pre*neg_rec/(neg_pre+neg_rec)
            
            new_meta=[pos_pre,pos_rec,neg_pre,neg_rec,pos_f,neg_f]
            new_aver=[ 1 if x>0 else 0 for x in new_meta]
            prf_mata=[sum(x) for x in zip(prf_mata,new_meta)]
            prf_aver=[sum(x) for x in zip(prf_aver,new_aver)]            
            
            if test_acc > best:
                best=test_acc

            
        print ('Epoch best performance---------', best)
        best_lst.append(best)
    print best_lst
    print ( 'Averaged testing accuracy---------',reduce(lambda x, y: x + y, best_lst) / len(best_lst))   
       
    
    last_prf=[x/float(y) if y!=0 else 0 for x,y in zip(prf_mata,prf_aver)]

        
    print ('Macro F1 score---------', 0.5*(last_prf[-1]+last_prf[-2]))
