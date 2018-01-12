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

def lengths_to_mask(lengths):
    return None


np.random.seed(7)







inputs = tf.placeholder(tf.float32,shape=(None, 35,300))

x1_shape = inputs.get_shape()

aspect_id = Input(dtype=np.float32,batch_shape=[x1_shape[0],x1_shape[1],1])

aspect_x1 = K.batch_dot(inputs,aspect_id, axes=[1, 1])


#conver the batch dot output to shape of x1(sequence output from lstm).
aspect_x2=K.squeeze(aspect_x1,2)

aspect_x3=tf.mul(aspect_x2,tf.reciprocal(tf.reduce_sum(aspect_id, 1)))


aspect_vec= K.repeat(aspect_x3, 35)



def td_lstm(lstm_outputs,indexes):
    batch_size = tf.shape(lstm_outputs)[0]
    length = tf.shape(lstm_outputs)[1]
    output_dims = 100#tf.shape(lstm_outputs)[2]
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

print ('sentence input shape  ',inputs.get_shape())








inputs_ = tf.concat(2,[aspect_vec,inputs])


x1 = LSTM(100,  return_sequences=True,W_regularizer=l2(0.01), consume_less='mem')(inputs_)

x2 = LSTM(100,  return_sequences=True,W_regularizer=l2(0.01), consume_less='mem',go_backwards=True)(inputs_)





vec_fw = td_lstm(x1,indexes_fw)
vec_bw = td_lstm(x2,indexes_bw)
vec_ = tf.concat(1,[vec_fw,vec_bw])




predictions= Dense(3,bias=False,activation='softmax')(vec_)



labels = tf.placeholder(tf.float32, shape=(None,3))



loss = tf.reduce_mean(categorical_crossentropy(labels, predictions))

train_step = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)

acc_value = accuracy(labels, predictions)

print '-------Model building complete--------'
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #load aspect mat info
    aspect_mat=pickle.load(open('../data/English/tc_td/2951l_aspectid.p','r'))

    multi=[]
##########################################
##########################################
    for i in range(aspect_mat.shape[0]):
        if np.sum(aspect_mat[i,:,:])==1:
            multi.append(i)
    index = [x for x in range(len(multi))]

# #restaurnat single-word subset    
#     test_ind=index[:804]
#     train_ind=index[805:]
# #laptop single-word subset    
#     test_ind=index[:351]
#     train_ind=index[352:]

##########################################
##########################################

    for i in range(aspect_mat.shape[0]):
        if np.sum(aspect_mat[i,:,:])>1:
            multi.append(i)
    index = [x for x in range(len(multi))]

#restaurnat multi-word subset    
#    test_ind=index[:316]
#    train_ind=index[317:]
# #laptop multi-word subset    
#     test_ind=index[:287]
#     train_ind=index[288:]

##########################################
##########################################
    aspect_mat=aspect_mat[multi]
    
    #load embeddings
    dataset=pickle.load(open('../data/English/tc_td/2951l_emb.p','r'))
    dataset=dataset[multi]  
    

    #load labels
    label0= codecs.open('../data/English/tc_td/l_label.txt','r','utf8').readlines()
    label=[]
    for e in label0:
        label.append(np.int(e.rstrip()))    

    label=np.take(label, multi)



    #load forward and backward aspect info
    indexes_f=pickle.load(open('../data/English/tc_td/lind_forward.p','r'))
    indexes_f=np.take(indexes_f, multi,axis=0)

    indexes_b=pickle.load(open('../data/English/tc_td/lind_backward.p','r'))
    indexes_b=np.take(indexes_b, multi,axis=0)






    num_train=np.int32(len(train_ind))
    num_test=np.int32(len(test_ind))
    num_minibatch=np.int32(len(train_ind)/50)
    
    best_lst=[]
    n=0
    for n in range(5):
        

        sess.run(tf.global_variables_initializer())
        best=0
        prf_mata=[0,0,0,0,0,0,0,0,0]
        prf_aver=[0,0,0,0,0,0,0,0,0]
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
    
           
            b=np.zeros((num_train,3))
            b[np.arange(num_train),y_train1]  =1
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
    
            
    
            c=np.zeros((num_test,3))
            c[np.arange(num_test),y_test1]  =1
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
    print ('Testing length', num_test)

