# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.models import Model
from keras.layers import Dropout, Activation, Bidirectional, Embedding,LSTM,Input, Dense,Wrapper, Recurrent, merge
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



inputs = tf.placeholder(tf.float32,shape=(None, 15,128))

x1_shape = inputs.get_shape()

aspect_id = Input(dtype=np.float32,batch_shape=[x1_shape[0],x1_shape[1],1])

aspect_x1 = K.batch_dot(inputs,aspect_id, axes=[1, 1])


aspect_x2=K.squeeze(aspect_x1,2)

aspect_x3=tf.mul(aspect_x2,tf.reciprocal(tf.reduce_sum(aspect_id, 1)))

aspect_vec= K.repeat(aspect_x3, 15)



def td_lstm(lstm_outputs,indexes):
    batch_size = tf.shape(lstm_outputs)[0]
    length = tf.shape(lstm_outputs)[1]
    output_dims = 32#tf.shape(lstm_outputs)[2]
    lstm_outputs = tf.reshape(lstm_outputs,[-1,output_dims])
    offsets = tf.range(batch_size)*length
    offsets = tf.expand_dims(offsets,1)
    indexes += offsets
    indexes = tf.reshape(indexes,[-1])
    outputs = tf.gather(lstm_outputs,indexes)
    outputs =tf.reshape(outputs,[-1,output_dims])
    return outputs

indexes_fw = tf.placeholder(tf.int32,shape=(None,1))
indexes_bw = tf.placeholder(tf.int32,shape=(None,1))









inputs_ = tf.concat(2,[aspect_vec,inputs])


x1 = LSTM(output_dim=32, input_dim=128, return_sequences=True,W_regularizer=l2(0.01), consume_less='mem')(inputs_)

x2 = LSTM(output_dim=32, input_dim=128, return_sequences=True,W_regularizer=l2(0.01), consume_less='mem',go_backwards=True)(inputs_)





vec_fw = td_lstm(x1,indexes_fw)
vec_bw = td_lstm(x2,indexes_bw)
vec_ = tf.concat(1,[vec_fw,vec_bw])




predictions= Dense(2,bias=False,activation='softmax')(vec_)





labels = tf.placeholder(tf.float32, shape=(None,2))



loss = tf.reduce_mean(binary_crossentropy(labels, predictions))

train_step = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)

acc_value = accuracy(labels, predictions)


print '-------Model building complete--------'
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    
    
    # load dataset embeddings, change according to dataset
    dataset=pickle.load(open('../data/Chinese/word/allembedding.p','r'))
    dataset_size=dataset.shape[0]
    

    # load dataset labels, change according to dataset
    label0= codecs.open('../data/Chinese/label/alllabel.txt','r','utf8').readlines()
    label=[]
    for e in label0:
        label.append(np.int(e.rstrip()))    

    # load aspect info
    aspect_mat=pickle.load(open('../data/Chinese/word/allaspectid.p','r'))
    indexes_f=pickle.load(open('../data/Chinese/word/tc-td-lstm/allind_forward.p','r'))
    indexes_b=pickle.load(open('../data/Chinese/word/tc-td-lstm/allind_backward.p','r'))


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
            y_train1=[]
            indexes_f_train1=[]
            indexes_b_train1=[]
            train_aspect_id=[]
            
            for e in train_ind:
                X_train1.append(dataset[e])
                y_train1.append(label[e])
                indexes_f_train1.append(indexes_f[e])
                indexes_b_train1.append(indexes_b[e])
                train_aspect_id.append(aspect_mat[e,:,:])
    
    
                
                
                
            X_train1=np.asarray(X_train1)    
            indexes_f_train1=np.asarray(indexes_f_train1)
            indexes_b_train1=np.asarray(indexes_b_train1)
            train_aspect_id=np.asarray(train_aspect_id)   
    
           
            b=np.zeros((len(y_train1),2))
            b[np.arange(len(y_train1)),y_train1]  =1
            y_train1=b
            
    
            
            
            X_test1=[]
            y_test1=[]
            indexes_f_test1=[]
            indexes_b_test1=[]
            test_aspect_id=[]
    
            for e in  test_ind:
                X_test1.append(dataset[e])
                y_test1.append(label[e])
                indexes_f_test1.append(indexes_f[e])
                indexes_b_test1.append(indexes_b[e]) 
                test_aspect_id.append(aspect_mat[e,:,:])
    
            
    
            c=np.zeros((len(y_test1),2))
            c[np.arange(len(y_test1)),y_test1]  =1
            y_test1=c    
                   
            start = 0
            end = 50
            for i in range(num_minibatch):
          
                
                X = X_train1[start:end]
                Y = y_train1[start:end]
                idd = train_aspect_id[start:end]
                forward=indexes_f_train1[start:end]
                backward=indexes_b_train1[start:end]
                
                
                start = end
                end = start + 50
                sess.run(train_step, feed_dict={inputs: X, indexes_fw:forward, indexes_bw:backward, aspect_id:idd, labels: Y, K.learning_phase(): 1})    
                
                

    
            # testing accuracy
            test_acc= acc_value.eval(feed_dict={inputs: X_test1, indexes_fw:indexes_f_test1, indexes_bw:indexes_b_test1, aspect_id:test_aspect_id, labels: y_test1,K.learning_phase(): 0})
            pre=predictions.eval(feed_dict={inputs: X_test1, indexes_fw:indexes_f_test1, indexes_bw:indexes_b_test1, aspect_id:test_aspect_id, labels: y_test1,K.learning_phase(): 0})
            

            
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
