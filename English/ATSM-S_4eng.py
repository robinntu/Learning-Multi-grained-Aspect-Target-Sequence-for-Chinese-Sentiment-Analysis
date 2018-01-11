# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.models import Model
from keras.layers import Dropout, Activation, Bidirectional, Embedding,LSTM,Input, Dense,Wrapper, Recurrent, merge,GRU,Convolution1D,Flatten,Reshape,MaxPooling1D
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

np.random.seed(7)

keras_learning_phase = tf.constant(True)


    
    
inputs = tf.placeholder(tf.float32,shape=(None, 35,300))

inputs_shape= inputs.get_shape()

lstm1 = LSTM(output_dim=75, return_sequences=True,W_regularizer=l2(0.01), consume_less='mem')(inputs)

aspect_id = Input(dtype=np.float32,batch_shape=[inputs_shape[0],inputs_shape[1],5])
x_batch_dot = K.batch_dot(inputs,aspect_id, axes=[1, 1])

batch_dot_shape =x_batch_dot.get_shape()


def aspect_layer(seq):
    vec=x_batch_dot[:,:,seq]
    vec2= K.repeat(vec, 35)   
    atten_inputs=tf.concat(2,[lstm1,vec2])
    atten_inputs=tf.cast(atten_inputs,tf.float32)

    gi= Dense(1, bias=True, activation='tanh')(atten_inputs)
    ai= Dense(1,bias=False,activation='softmax')(gi)
    atten_output = K.batch_dot(atten_inputs,ai, axes=[1, 1])
    atten_output=K.squeeze(atten_output,2)
     
    return atten_output





# aspect from 1 to 5
atten_output1=aspect_layer(0)
atten_output2=aspect_layer(1)
atten_output3=aspect_layer(2)
atten_output4=aspect_layer(3)
atten_output5=aspect_layer(4)








# construct input sequence for target sequence learning
lstm_input= tf.concat(1,[tf.expand_dims(atten_output1, 1),tf.expand_dims(atten_output2, 1)])
lstm_input1=tf.concat(1,[lstm_input,tf.expand_dims(atten_output3, 1)])
lstm_input2=tf.concat(1,[lstm_input1,tf.expand_dims(atten_output4, 1)])
lstm_input3=tf.concat(1,[lstm_input2,tf.expand_dims(atten_output5, 1)])

var0= LSTM(output_dim=100,  activation='tanh',return_sequences=False,W_regularizer=l2(0.01), inner_activation='sigmoid')(lstm_input3)



var3= Dropout(0.5)(var0)

#Softmax layer
predictions= Dense(3,bias=False,activation='softmax')(var3)


#data labels
labels = tf.placeholder(tf.float32, shape=(None,3))

loss = tf.reduce_mean(categorical_crossentropy(labels, predictions))

train_step = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)

acc_value = accuracy(labels, predictions)

print '-------Model building complete--------'
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#Load data from laptop domain from SemEval 2014

    #single/multi training
    seqtr=pickle.load(open('../data/English/single_multi_split_lst/ltrain_single_multi_lst.py','r'))
    #single/multi training
    #[0] is single case
    #[1] is multi case
    multitr=seqtr[1]

    #single/multi testing
    seqte=pickle.load(open('../data/English/single_multi_split_lst/ltest_single_multi_lst.py','r'))
    #single/multi testing
    multite=seqte[1]


    # load dataset word embedding 
    dataset_train=pickle.load(open('../data/English/dataset/ltrain_emb.py','r'))
    dataset_train=dataset_train[multitr]

    dataset_test=pickle.load(open('../data/English/dataset/ltest_emb.py','r'))
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
       
       
       
       
    # load aspect info of the dataset
    aspect_mat_train=pickle.load(open('../data/English/dataset/ltrain_aspectid.py','r'))
    aspect_mat_train=aspect_mat_train[multitr]


    aspect_mat_test=pickle.load(open('../data/English/dataset/ltest_aspectid.py','r'))
    aspect_mat_test=aspect_mat_test[multite]
######################################################################################################################################
#######################################################################################################################################
# #######################################################################################################################################
# #Load data from restaurant domain from SemEval 2014

#     #single/multi training
#     seqtr=pickle.load(open('../data/English/single_multi_split_lst/rtrain_single_multi_lst.py','r'))
#     #single/multi training
#     #[0] is single case
#     #[1] is multi case
#     #single/multi testing
#     multitr=seqtr[1]
    
#     #single/multi testing
#     seqte=pickle.load(open('../data/English/single_multi_split_lst/rtest_single_multi_lst.py','r'))
#     #single/multi testing
#     multite=seqte[1]
    
    
    
#     # load dataset word embedding 
#     dataset_train=pickle.load(open('../data/English/dataset/rtrain_emb.py','r'))
#     dataset_train=dataset_train[multitr]
    
#     dataset_test=pickle.load(open('../data/English/dataset/rtest_emb.py','r'))
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
        
        
        
        
#     # load aspect info of the dataset
#     aspect_mat_train=pickle.load(open('../data/English/dataset/rtrain_aspectid.py','r'))
#     aspect_mat_train=aspect_mat_train[multitr]
    

#     aspect_mat_test=pickle.load(open('../data/English/dataset/rtest_aspectid.py','r'))
#     aspect_mat_test=aspect_mat_test[multite]

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
            print 'trainning for ' +'Epoch'+ str(epoch)          

            X_train1=[]
            train_aspect_id=[]
            y_train1=[]
            for e in train_ind:
                X_train1.append(dataset_train[e])
                train_aspect_id.append(aspect_mat_train[e,:,:])
                y_train1.append(label_train[e]) 
              
                 
            
            X_train1=np.asarray(X_train1)    
            train_aspect_id=np.asarray(train_aspect_id)  
    
    
            
            b=np.zeros((num_tr,3))
            b[np.arange(num_tr),y_train1]  =1
            y_train1=b
            
    
    
            
            
            X_test1=[]
            y_test1=[]
            test_aspect_id=[]
            for e in  test_ind:
                X_test1.append(dataset_test[e])
                test_aspect_id.append(aspect_mat_test[e,:,:])
                y_test1.append(label_test[e])
            
            
            c=np.zeros((num_te,3))
            c[np.arange(num_te),y_test1]  =1
            y_test1=c    
    
        
    
            
            start = 0
            end = 50
            for i in range(num_tr/int(50)):
                
                X = X_train1[start:end]
                Y = y_train1[start:end]
                idd = train_aspect_id[start:end]
                start = end
                end = start + 50
                sess.run(train_step, feed_dict={inputs: X,  aspect_id:idd, labels: Y, K.learning_phase(): 1})    
                
                
            Loss = str(sess.run(loss, feed_dict={inputs: X_train1,  aspect_id:train_aspect_id, labels: y_train1,K.learning_phase(): 0}))
            
            # trainning loss
            print ('Training Loss -------',Loss)
            
            

    
            # testing accuracy
            test_acc= acc_value.eval(feed_dict={inputs: X_test1,  aspect_id:test_aspect_id, labels: y_test1,K.learning_phase(): 0})
            
            
            print test_acc
        

            pre=predictions.eval(feed_dict={inputs: X_test1,  aspect_id:test_aspect_id, labels: y_test1,K.learning_phase(): 0})
    
            
            #Compute precision, recall, F1
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












