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



inputs_noas=tf.placeholder(tf.float32,shape=(None, 35,300))




aspect_id = Input(dtype=np.float32,batch_shape=[inputs_shape[0],inputs_shape[1],1])




aspect_x1 = K.batch_dot(inputs,aspect_id, axes=[1, 1])

aspect_x2=K.squeeze(aspect_x1,2)

aspect_x3=tf.mul(aspect_x2,tf.reciprocal(tf.reduce_sum(aspect_id, 1)))




linear= Dense(300,bias=False,activation='linear')(aspect_x3)






aspect_x4= K.repeat(aspect_x3, 35)
att_inputs=merge([inputs_noas,aspect_x4], mode='concat')


gi= Dense(1, bias=True, activation='tanh')(att_inputs)

ai= tf.nn.softmax(gi, dim=1, name=None)


vec = K.batch_dot(inputs_noas,ai, axes=[1, 1])
vec=K.squeeze(vec,2)




x2=merge([vec,linear], mode='sum')



for i in range(8):  # number of hops
    
    aspect_x4= K.repeat(x2, 35)
    att_inputs=merge([inputs_noas,aspect_x4], mode='concat')
    

    gi= Dense(1, bias=True, activation='tanh')(att_inputs)
    
    ai= tf.nn.softmax(gi, dim=1, name=None)

    
    vec = K.batch_dot(inputs_noas,ai, axes=[1, 1])
    vec=K.squeeze(vec,2)
    
    
    
    x2=merge([vec,linear], mode='sum')
        
    













predictions= Dense(3,bias=False,activation='softmax')(x2)



labels = tf.placeholder(tf.float32, shape=(None,3))


loss = tf.reduce_mean(categorical_crossentropy(labels, predictions))

train_step = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)

acc_value = accuracy(labels, predictions)


print '-------Model building complete--------'
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    #single/multi testing
    seq=pickle.load(open('../data/English/memnet/all_single_multi_lst.p','r'))


#loading and filtering dataset
##############################
##############################
##############################    
    multi=seq[0]
    index = [x for x in range(len(multi))]
#restaurnat single-word subset    
    test_ind=index[1818:2621]
    train_ind=index[2622:]

#laptop single-word subset    
    test_ind=index[:351]
    train_ind=index[352:1817]
##############################
##############################
##############################

    multi=seq[1]
    index = [x for x in range(len(multi))]
#restaurnat multi-word subset    
    test_ind=index[1135:1450]
    train_ind=index[1451:]
#laptop multi-word subset    
    test_ind=index[:287]
    train_ind=index[288:1134]
##############################
##############################
##############################
    num_te=len(test_ind)



    dataset=pickle.load(open('../data/English/memnet/all_sent_emb.p','r'))
    
    #single/multi testing
    dataset=dataset[multi]
  
    print dataset.shape

 
    noasdt=pickle.load(open('../data/English/memnet/all_notar_sent_emb.p','r'))
    #single/multi testing
    noasdt=noasdt[multi]

   
    label0= codecs.open('../data/English/memnet/all_label.txt','r','utf8').readlines()
    
    label=[]
    for e in label0:
        label.append(np.int(e.rstrip()))    
    #single/multi testing
    label=np.take(label, multi)


   
    aspect_mat=pickle.load(open('../data/English/memnet/all_aspectid.p','r'))
    #single/multi testing
    aspect_mat=aspect_mat[multi]





    best_lst=[]
    n=0
    for n in range(5):
        
       
    
        random.shuffle(train_ind)
        random.shuffle(test_ind)    
    
        sess.run(tf.global_variables_initializer())
        best=0
        prf_mata=[0,0,0,0,0,0,0,0,0]
        prf_aver=[0,0,0,0,0,0,0,0,0]
        for epoch in range(50):
            print 'train_ning for ' +'Epoch'+ str(epoch)          

        
            X_train1=[]
            train_aspect_id=[]
            y_train1=[]
            noastrain=[]
            for e in train_ind:
                X_train1.append(dataset[e])
                train_aspect_id.append(aspect_mat[e,:,:])
                y_train1.append(label[e])
                noastrain.append(noasdt[e])
                
            X_train1=np.asarray(X_train1)    
            train_aspect_id=np.asarray(train_aspect_id)   

            
            b=np.zeros((len(train_ind),3))
            b[np.arange(len(train_ind)),y_train1]  =1
            y_train1=b
            

            
            
            X_test1=[]
            y_test1=[]
            test_aspect_id=[]
            noastest=[]
            for e in  test_ind:
                X_test1.append(dataset[e])
                test_aspect_id.append(aspect_mat[e,:,:])
                y_test1.append(label[e])
                noastest.append(noasdt[e])
            

            c=np.zeros((len(test_ind),3))
            c[np.arange(len(test_ind)),y_test1]  =1
            y_test1=c    

        
    
            
            start = 0
            end = 50
            for i in range(len(train_ind)/50):
         
                
                X = X_train1[start:end]
                Y = y_train1[start:end]
                idd = train_aspect_id[start:end]
                noas = noastrain[start:end]
                start = end
                end = start + 50
                sess.run(train_step, feed_dict={inputs: X,inputs_noas:noas,  aspect_id:idd, labels: Y, K.learning_phase(): 1})    
                
                
            Loss = str(sess.run(loss, feed_dict={inputs: X_train1, inputs_noas: noastrain, aspect_id:train_aspect_id, labels: y_train1,K.learning_phase(): 0}))
            
            # trainning loss
            print ('Training Loss -------',Loss)
            
            

    
            # testing accuracy
            test_acc= acc_value.eval(feed_dict={inputs: X_test1, inputs_noas: noastest, aspect_id:test_aspect_id, labels: y_test1,K.learning_phase(): 0})
        
            pre=predictions.eval(feed_dict={inputs: X_test1, inputs_noas: noastest, aspect_id:test_aspect_id, labels: y_test1,K.learning_phase(): 0})
            
            
            


    
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
