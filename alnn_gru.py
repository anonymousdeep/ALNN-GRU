import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np






class ALNNLayer(tf.keras.layers.Layer):
    
    def __init__(self,prior_hours,init_time=0,time_space=13,type_distance="abs"):
        super(ALNNLayer, self).__init__()
        self.prior_hours = prior_hours
        self.init_time = init_time
        self.time_space=time_space
        self.maxpool=layers.MaxPooling1D(2)
        self.aggregate_loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.nr_ref_time_points=time_space
        self.type_distance=type_distance
        
        
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
        
        
        if(self.type_distance=="abs"):
            self.diastance=tf.abs(self.t-tf.cast(self.ref_time,tf.float32))
        else:
            self.diastance=tf.square(self.t-tf.cast(self.ref_time,tf.float32))
            
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
    
    def __init__(self,
                 max_time,
                 init_time,
                 time_interval,
                 type_distance,
                 gru_unit=168,
                 gru_dropout=0.0,
                 pseudo_latent_dropout=0.0):
        
        
        super(ALNN_GRU, self).__init__()
        
        self.max_time=max_time
        self.init_time=init_time
        self.time_interval=time_interval
        self.type_distance=type_distance
        self.gru_unit=gru_unit
        self.gru_dropout=gru_dropout
        self.pseudo_latent_dropout=pseudo_latent_dropout
        
        
        self.ALNNLayer=ALNNLayer(self.max_time,
                                 self.init_time,
                                 self.time_interval,
                                 self.type_distance)
        
        
        self.gru=layers.GRU(self.gru_unit,dropout=self.gru_dropout)
        self.dense=layers.Dense(1,activation='sigmoid')
        self.dropout_1=layers.Dropout(self.pseudo_latent_dropout)


    def call(self, inputs,training=None):
        self.x=tf.cast(inputs[0],tf.float32)
        self.t=tf.cast(inputs[1],tf.float32)
        self.m=tf.cast(inputs[2],tf.float32)
        self.d_t=tf.cast(inputs[3],tf.float32)


        self.lattent_data=self.ALNNLayer([self.x,self.t,self.m,self.d_t])

        if training:          
            self.lattent_data=self.dropout_1(self.gru(self.lattent_data),training=training)   
        else:
            self.lattent_data=self.gru(self.lattent_data)

        return self.dense(self.lattent_data)
    
    def get_config(self):
        config = super(ALNN_GRU, self).get_config()
        self.max_time=max_time
        self.init_time=init_time
        self.time_interval=time_interval
        self.type_distance=type_distance
        self.gru_unit=gru_unit
        self.gru_dropout=gru_dropout
        self.pseudo_latent_dropout=pseudo_latent_dropout
        
        config.update({"max_time": self.max_time})
        config.update({"init_time": self.init_time})
        config.update({"time_interval": self.time_interval})
        config.update({"type_of_distance": self.type_distance})
        config.update({"gru_unit": self.gru_unit})
        config.update({"gru_dropout": self.gru_dropout})
        config.update({"pseudo_latent_dropout": self.pseudo_latent_dropout})
        return config
