import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np

class ALNNLayer(tf.keras.layers.Layer):
    
    def __init__(self,prior_hours,init_time=0,time_space=13):
        super(ALNNLayer, self).__init__()
        self.prior_hours = prior_hours
        self.init_time = init_time
        self.time_space=time_space
        self.maxpool=layers.MaxPooling1D(2)
        self.aggregate_loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.nr_ref_time_points=time_space
        
        
#         if((self.prior_hours%self.time_space)!=0):
#             raise Exception(f'{self.time_space}  must be a multiple of {self.prior_hours}.')
            
        self.ref_time=np.linspace(init_time,self.prior_hours,self.nr_ref_time_points)
        self.ref_time=self.ref_time.reshape(self.nr_ref_time_points,1,1)
        
        self.dropout_1=layers.Dropout(0.05)
        self.dropout_2=layers.Dropout(0.05)

        

    def build(self, input_shape):
            
        self.axis_2=input_shape[0][1]
        self.axis_3=input_shape[0][2]
        
        self.alpha = self.add_weight(shape=(self.nr_ref_time_points,1,1),
                                 initializer='random_normal',
                                 name='alpha',
                                 dtype='float32',
                                 trainable=True)
        
        self.w_v = self.add_weight(shape=(self.nr_ref_time_points,self.axis_2,input_shape[0][2]),
                                 initializer='random_normal',
                                 name='w_inyensity',
                                 dtype='float32',
                                 trainable=True)
        
        
        self.w_t = self.add_weight(shape=(self.nr_ref_time_points,self.axis_2,input_shape[0][2],4),
                                 initializer='random_normal',
                                 name='w_tempo',
                                 dtype='float32',
                                 trainable=True)
        
        
        self.b_v= self.add_weight(shape=(self.nr_ref_time_points,1,input_shape[0][2]),
                                 initializer='random_normal',
                                 name='bias_intensity',
                                 dtype='float32',
                                 trainable=True)
        
        self.b_t = self.add_weight(shape=(self.nr_ref_time_points,self.axis_2, self.axis_3,1),
                                 initializer='random_normal',
                                 name='bias_tempo',
                                 dtype='float32',
                                 trainable=True)
        
    def call(self, inputs,training=None):
        self.X=inputs[0]
        self.T=inputs[1]
        self.M=inputs[2]
        self.DT=inputs[3]
                
        self.x=tf.tile(self.X[:,None,:,:],[1,self.nr_ref_time_points,1,1])
        self.t=tf.tile(self.T[:,None,:,:],[1,self.nr_ref_time_points,1,1])
        self.m=tf.tile(self.M[:,None,:,:],[1,self.nr_ref_time_points,1,1])
        self.dt=tf.tile(self.DT[:,None,:,:],[1,self.nr_ref_time_points,1,1])
        
        

        self.diastance=tf.abs(self.t-tf.cast(self.ref_time,tf.float32))                      
        self.kernel=tf.exp(-tf.cast(tf.nn.relu(self.alpha),tf.float32)*self.diastance)
        self.intensity=tf.nn.relu(self.x*self.kernel)
        
        self.x_s=tf.reshape(self.x,[-1,self.nr_ref_time_points,self.axis_2, self.axis_3,1])
        self.dt=tf.reshape(self.dt,[-1,self.nr_ref_time_points,self.axis_2, self.axis_3,1])
        self.intensity_s=tf.reshape(self.intensity,[-1,self.nr_ref_time_points,self.axis_2, self.axis_3,1])
        self.m_s=tf.reshape(self.m,[-1,self.nr_ref_time_points,self.axis_2, self.axis_3,1])
        
        
        
        if training:
            self.lattent_x=self.dropout_1(tf.nn.relu(tf.reduce_sum(self.w_t*tf.concat([self.x_s,self.dt,self.intensity_s,self.m_s],4)+self.b_t,4)),training=training)
            self.lattent_x=self.dropout_2(tf.nn.relu(tf.reduce_sum(self.w_v*self.lattent_x + self.b_v,2)),training=training)
        else:
            self.lattent_x=tf.nn.relu(tf.reduce_sum(self.w_t*tf.concat([self.x_s,self.dt,self.intensity_s,self.m_s],4)+self.b_t,4))
            self.lattent_x=tf.nn.relu(tf.reduce_sum(self.w_v*self.lattent_x + self.b_v,2))



        return self.lattent_x
    
    def get_config(self):
        config = super(ALNNLayer, self).get_config()
        config.update({"prior_hours": self.prior_hours})
        config.update({"init_time": self.init_time})
        return config
    
    
class ALNN_GRU(keras.Model):
    
    def __init__(self,max_hour,init_time,time_interval,voc_size,unit_embedding,input_length):
        super(ALNN_GRU, self).__init__()
        
        self.max_hour=max_hour
        self.init_time=init_time
        self.time_interval=time_interval
        self.voc_size=voc_size
        self.unit_embedding=unit_embedding
        self.input_length=input_length
        
        self.ALNNLayer=ALNNLayer(self.max_hour,init_time=self.init_time,time_space=self.time_interval)
        self.gru=layers.GRU(168,dropout=0.0000)
        self.gru2=layers.GRU(50,dropout=0.2)
        
        self.dense2=layers.Dense(50,activation='relu')
        self.dense=layers.Dense(1,activation='sigmoid')
        
        self.dropout_1=layers.Dropout(0.0)
        
        #0.5/0.7
        self.dropout_2=layers.Dropout(0.1)
        self.dropout_3=layers.Dropout(0.9)
        
        self.embedding = layers.Embedding(self.voc_size,self.unit_embedding,input_length=self.input_length,mask_zero=True)
        self.embedding_2 = layers.Embedding(2,self.unit_embedding,input_length=self.input_length,mask_zero=True)

        
        self.mse=tf.keras.losses.MeanSquaredError()
        
        self.conv1d_diag=layers.Conv1D(20,10,padding='valid',activation='relu')
        self.max_pooling1d_diag=layers.MaxPooling1D(pool_size=20, strides=None, padding='valid')
        self.flatten_diag_1=layers.Flatten()
        
        self.attention=Attention(4,40)


    def call(self, inputs,training=None):
        self.x=tf.cast(inputs[0],tf.float32)
        self.t=tf.cast(inputs[1],tf.float32)
        self.m=tf.cast(inputs[2],tf.float32)
        self.d_t=tf.cast(inputs[3],tf.float32)
        self.histo=tf.cast(inputs[4],tf.float32)


        self.lattent_data=self.ALNNLayer([self.x,self.t,self.m,self.d_t])

        
        
        if training:
            self.parent=tf.cast(inputs[5],tf.float32)
            self.relation=tf.cast(inputs[6],tf.float32)
            
            self.n_parent=tf.cast(inputs[7],tf.float32)
            self.n_relation=tf.cast(inputs[8],tf.float32)
            
            
            self.parent_=self.dropout_2(self.embedding(self.parent))
            self.n_parent_=self.dropout_2(self.embedding(self.n_parent))
            
            self.relation_=self.embedding_2(self.relation)
            self.n_relation_=self.embedding_2(self.n_relation)
            
                        
            self.lattent_data=self.dropout_1(self.gru(self.lattent_data),training=training)
            self.h=self.dropout_2(self.embedding(self.histo))
            
            self.n_error=tf.square(tf.reduce_sum(tf.pow((self.h+self.n_relation_-self.n_parent_),2),2))
            
            self.error=tf.nn.relu(tf.square(tf.reduce_sum(tf.pow((self.h+self.relation_-self.parent_),2),2))-self.n_error+tf.cast(0.005,tf.float32))
                        
            self.add_loss(tf.reduce_sum(self.error))
            
        else:
            self.lattent_data=self.gru(self.lattent_data)
            self.h=self.embedding(self.histo)
            


        self.h=tf.reduce_sum(self.h,1)

        
        if training:
            self.h=self.dropout_3(self.h)


        self.lattent_data=tf.concat([self.lattent_data,self.h],1)

            
        return self.dense(self.lattent_data)
