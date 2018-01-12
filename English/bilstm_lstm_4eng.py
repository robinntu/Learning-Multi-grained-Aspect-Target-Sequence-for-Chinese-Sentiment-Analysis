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
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
from keras.regularizers import l2
import random



import codecs
import cPickle as pickle
import time
time0=time.time()



np.random.seed(7)

keras_learning_phase = tf.constant(True)


inputs_w = tf.placeholder(tf.float32,shape=(None, 35,300))


inputs_w_shape= inputs_w.get_shape()



#lstm (comment the bilstm below)
#lstm1_w = LSTM(output_dim=32,activation='tanh', return_sequences=False,W_regularizer=l2(0.01),inner_activation='hard_sigmoid', consume_less='mem')(inputs_w)

#blstm (comment the lstm above)
lstm1_w= Bidirectional(LSTM(output_dim=50,activation='tanh', return_sequences=False,W_regularizer=l2(0.01),inner_activation='tanh', consume_less='mem'),merge_mode='concat')(inputs_w)


var_w= Dropout(0.5)(lstm1_w)



predictions= Dense(3,bias=False,activation='softmax')(var_w)


labels = tf.placeholder(tf.float32, shape=(None,3))



loss = tf.reduce_mean(categorical_crossentropy(labels, predictions))

train_w_step = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)

acc_value = accuracy(labels, predictions)


print '-------Model building complete--------'    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#Load data from laptop domain from SemEval 2014

    #single/multi training
    seqtr=pickle.load(open('../data/English/single_multi_split_lst/ltrain_single_multi_lst.p','r'))
    #single/multi training
    #[0] is single case
    #[1] is multi case
    multitr=seqtr[1]

    #single/multi testing
    seqte=pickle.load(open('../data/English/single_multi_split_lst/ltest_single_multi_lst.p','r'))
    #single/multi testing
    multite=seqte[1]


    # load dataset word embedding 
    dataset_train=pickle.load(open('../data/English/dataset/ltrain_emb.p','r'))
    dataset_train=dataset_train[multitr]

    dataset_test=pickle.load(open('../data/English/dataset/ltest_emb.p','r'))
    dataset_test=dataset_test[multite]


    # load sentiment labels of dataset
    label0= codecs.open('../data/English/label/ltrain_label.txt','r','utf8').readlines()
    label_train=[]
    for e in label0:
       label_train.append(np.int(e.rstrip()))  
    label_train=np.take(label_train, multitr)

    label0= codecs.open('../data/English/label/ltest_label.txt','r','utf8').readlines()
    label_test=[]
    for e in label0:
       label_test.append(np.int(e.rstrip()))  
    label_test=np.take(label_test, multite)
        
######################################################################################################################################
#######################################################################################################################################
# #######################################################################################################################################
# #Load data from restaurant domain from SemEval 2014

#     #single/multi training
#     seqtr=pickle.load(open('../data/English/single_multi_split_lst/rtrain_single_multi_lst.p','r'))
#     #single/multi training
#     #[0] is single case
#     #[1] is multi case
#     #single/multi testing
#     multitr=seqtr[1]
    
#     #single/multi testing
#     seqte=pickle.load(open('../data/English/single_multi_split_lst/rtest_single_multi_lst.p','r'))
#     #single/multi testing
#     multite=seqte[1]
    
    
    
#     # load dataset word embedding 
#     dataset_train=pickle.load(open('../data/English/dataset/rtrain_emb.p','r'))
#     dataset_train=dataset_train[multitr]
    
#     dataset_test=pickle.load(open('../data/English/dataset/rtest_emb.p','r'))
#     dataset_test=dataset_test[multite]


#     # load sentiment labels of dataset
#     label0= codecs.open('../data/English/label/rtrain_label.txt','r','utf8').readlines()
#     label_train=[]
#     for e in label0:
#         label_train.append(np.int(e.rstrip()))  
#     label_train=np.take(label_train, multitr)


#     # this is the sentiment labels of dataset
#     label0= codecs.open('../data/English/label/rtest_label.txt','r','utf8').readlines()
#     label_test=[]
#     for e in label0:
#         label_test.append(np.int(e.rstrip()))  
#     label_test=np.take(label_test, multite)

# ######################################################################################################################################
######################################################################################################################################
######################################################################################################################################    


    num_tr=len(seqtr[1])
    num_te=len(seqte[1])


##################
    best_lst=[]
    n=0
    for n in range(5):
        
       
    
        train_ind = [x for x in range(num_tr)]
        random.shuffle(train_ind)


        test_ind = [x for x in range(num_te)]
        random.shuffle(test_ind)
    
        sess.run(tf.global_variables_initializer())
        best=0
        prf_mata=[0,0,0,0,0,0,0,0,0]
        prf_aver=[0,0,0,0,0,0,0,0,0]
        for epoch in range(50):
            print 'train_nning for ' +'Epoch'+ str(epoch)          

            X_train1=[]
            y_train1=[]
            for e in train_ind:
                X_train1.append(dataset_train[e])
                y_train1.append(label_train[e]) 
              
                 
            
            X_train1=np.asarray(X_train1)    
    
    
            
            b=np.zeros((num_tr,3))
            b[np.arange(num_tr),y_train1]  =1
            y_train1=b
            
    
    
            
            
            X_test1=[]
            y_test1=[]
            for e in  test_ind:
                X_test1.append(dataset_test[e])
                y_test1.append(label_test[e])
            
            
            c=np.zeros((num_te,3))
            c[np.arange(num_te),y_test1]  =1
            y_test1=c    
    
        
    
            
            start = 0
            end = 50
            for i in range(num_tr/int(50)):
      
                
                X = X_train1[start:end]
                Y = y_train1[start:end]
                start = end
                end = start + 50
                sess.run(train_w_step, feed_dict={inputs_w: X,  labels: Y, K.learning_phase(): 1})    
                
                
            Loss = str(sess.run(loss, feed_dict={inputs_w: X_train1,   labels: y_train1,K.learning_phase(): 0}))
            
            # trainning loss
            print ('Training Loss -------',Loss)
            
            
    
            # testing accuracy
            test_acc= acc_value.eval(feed_dict={inputs_w: X_test1,  labels: y_test1,K.learning_phase(): 0})
            
            pre=predictions.eval(feed_dict={inputs_w: X_test1,  labels: y_test1,K.learning_phase(): 0})
    
            
  
            predict=[]
            for entry in pre:
                if entry[0]>entry[1] and entry[0]>entry[2]:
                    predict.append(-1)
                elif entry[1]>entry[0] and entry[1]>entry[2]:
                    predict.append(0)
                else:
                    predict.append(1)
                    
                    
            truth=[]
            for entry in y_test1:
                     
                if entry[0]>entry[1] and entry[0]>entry[2]:
                    truth.append(-1)
                elif entry[1]>entry[0] and entry[1]>entry[2]:
                    truth.append(0)
                else:
                    truth.append(1)        
    
            
            pp_denominator=0
            pp_numerator=0
            
            np_denominator=0
            np_numerator=0
            
            zp_denominator=0
            zp_numerator=0
            
            pr_denominator=0
            pr_numerator=0
            
            nr_denominator=0
            nr_numerator=0
                   
            zr_denominator=0
            zr_numerator=0
            
            
            
            
            pos_pre=0
            pos_rec=0
            neg_pre=0
            neg_rec=0
            zer_pre=0
            zer_rec=0
            pos_f=0
            neg_f=0
            zer_f=0
            
            
            for t, p in zip(truth, predict):
                if p>0:
                    #pos precision
                    pp_denominator+=1
                    if t>0:
                        pp_numerator+=1
                    
                elif p<0:
                    #neg precision
                    np_denominator+=1
                    if t<0:
                        np_numerator+=1
                     
                else:
                    #zero(neutral) precision
                    zp_denominator+=1
                    if t==0:
                        zp_numerator+=1                     
                    
                if t>0:
                    #pos recall
                    pr_denominator+=1
                    if p>0:
                        pr_numerator+=1
                elif t<0:
                    #neg recall
                    nr_denominator+=1
                    if p<0:
                        nr_numerator+=1
                        
                else:
                    #zero(neutral) recall
                    zr_denominator+=1
                    if p==0:
                        zr_numerator+=1                    
                        
                        
                    
            if pp_denominator!=0:
                pos_pre=pp_numerator/float(pp_denominator)
            if pr_denominator !=0:
                pos_rec=pr_numerator/float(pr_denominator)
            if np_denominator !=0:
                neg_pre=np_numerator/float(np_denominator)
            if nr_denominator !=0:
                neg_rec=nr_numerator/float(nr_denominator)       
            if zp_denominator !=0:
                zer_pre=zp_numerator/float(zp_denominator)
            if zr_denominator !=0:
                zer_rec=zr_numerator/float(zr_denominator)           
                
                
                
                
            
            if (pos_pre+pos_rec)!=0:
                pos_f= 2*pos_pre*pos_rec/(pos_pre+pos_rec)
            if (neg_pre+neg_rec)!=0:
                neg_f= 2*neg_pre*neg_rec/(neg_pre+neg_rec)
            if (zer_pre+zer_rec)!=0:
                zer_f= 2*zer_pre*zer_rec/(zer_pre+zer_rec)
                
                
                
                
            new_meta=[pos_pre,pos_rec,neg_pre,neg_rec,zer_pre,zer_rec,pos_f,neg_f,zer_f]
            new_aver=[ 1 if x>0 else 0 for x in new_meta]
            prf_mata=[sum(x) for x in zip(prf_mata,new_meta)]
            prf_aver=[sum(x) for x in zip(prf_aver,new_aver)]                     
            
            
            
           
            
            if test_acc > best:
                best=test_acc
      

      
        print ('Epoch best perfromance---------', best)
        best_lst.append(best)
    print best_lst
    print ( 'Averaged testing accuracy---------',reduce(lambda x, y: x + y, best_lst) / len(best_lst))   
    
        
    last_prf=[x/float(y) if y!=0 else 0 for x,y in zip(prf_mata,prf_aver)]
        
    print ('Macro F1 score---------', 0.333*(last_prf[-1]+last_prf[-2]+last_prf[-3]))        
    print ('Testing length', num_te)
