import os
import numpy as np
import cv2
import tensorflow as tf
from omniloader import OmniglotLoader as og

class MatchNet():

    eps = 1e-10
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    n_supports = tf.placeholder(tf.int32)
    n_way = tf.placeholder(tf.int32)
    x_i = tf.placeholder(tf.float32, shape=[None, None, og.im_size, og.im_size, og.im_channel])
    y_i_idx = tf.placeholder(tf.int32, shape=[None, None]) # batch size, n_support
    x_hat = tf.placeholder(tf.float32, shape=[None, og.im_size, og.im_size, og.im_channel])
    y_hat_idx = tf.placeholder(tf.int32, shape=[None,]) # batch size
    
    y_i = tf.one_hot(y_i_idx, n_way)
    y_hat = tf.one_hot(y_hat_idx, n_way)

    def convnet_encoder(self, inputs, reuse=False):
        k_sz = 3
        f_sz = 64
        c_sz = 1
        n_stack = 4
        layer = inputs
        for l in range(n_stack):
            with tf.variable_scope('conv_{}'.format(l)):
                filters = tf.get_variable('filter', [k_sz, k_sz, c_sz, f_sz], initializer=tf.constant_initializer(0.5))
                beta = tf.get_variable('BN_beta', [f_sz], initializer=tf.constant_initializer(0.0))
                gamma = tf.get_variable('BN_gamma', [f_sz], initializer=tf.constant_initializer(1.0))
                c_sz = f_sz
                Z = tf.nn.conv2d(layer, filters, strides=[1,1,1,1], padding='SAME')
                mu, sig = tf.nn.moments(Z, [0,1,2])
                Z_tild = tf.nn.batch_normalization(Z, mu, sig, beta, gamma, self.eps)
                activ = tf.nn.relu(Z_tild)
                layer = tf.nn.max_pool(activ, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        return layer

    def __init__(self, share_encoder=True):
        self.sharing = share_encoder
        scope = 'image_encoder'
        with tf.variable_scope(scope):
            self.x_hat_encoded = self.convnet_encoder(self.x_hat)
        
        if not self.sharing:
            scope = 'support_set_encoder'
        
        # self.x_i_encoded = []
        # with tf.variable_scope(scope, reuse=self.sharing):
        #     for i in range(5): # range(self.n_supports):
        #         i_encoded = self.convnet_encoder(self.x_i[:,i,:,:,:])
        #         self.x_i_encoded.append(i_encoded)

        # with tf.variable_scope(scope, reuse=self.sharing):
        #     i = tf.constant(0)
        #     self.x_i_loop = tf.constant(0, shape=[1, 16, 1, 1, 64], dtype=tf.float32)
        #     for_cond = lambda i, x_i_loop : i < self.n_supports
        #     # body = lambda i, x_i_loop : [i+1, x_i_loop.append(self.convnet_encoder(self.x_i[:,i,:,:,:]))]
        #     def body(i, x_i_loop):
        #         self.x_i_loop = tf.concat( [self.x_i_loop, tf.expand_dims(self.convnet_encoder(self.x_i[:,i,:,:,:]),0) ] , axis=0)
        #         return [tf.add(i,1), self.x_i_loop]
        #     _, self.x_i_encoded = tf.while_loop(for_cond, body, loop_vars=[i, self.x_i_loop], shape_invariants=[i.get_shape(), tf.TensorShape([None,None,1,1,64])])

        with tf.variable_scope(scope, reuse=self.sharing):
            i = tf.constant(0)
            x_i_loop = tf.constant(0, shape=[1,16,1,1,64], dtype=tf.float32)
            for_cond = lambda i, x_i_loop : i < self.n_supports
            body = lambda i, x_i_loop : [i+1, tf.concat( [x_i_loop, tf.expand_dims(self.convnet_encoder(self.x_i[:,i,:,:,:]),0)], axis=0 )]
            _, self.x_i_encoded = tf.while_loop(for_cond, body, loop_vars=[i,x_i_loop], shape_invariants=[i.get_shape(), tf.TensorShape([None,None,1,1,64])] )


if __name__ == '__main__':
    
    loader = og(0)
    model = MatchNet()    
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    print(session.run(tf.report_uninitialized_variables()))

    while True:
        batch_size = 16
        N_way = 10
        k_shot = 1
        x_support, y_support, x_query, y_query = loader.getTrainSample(batch_size, N_way, k_shot)
        [ x_h_, x_i_ ] = session.run([model.x_hat_encoded, 
                model.x_i_encoded], feed_dict={
                model.n_supports: N_way*k_shot, model.n_way: N_way,
                model.x_i: x_support, model.y_i_idx: y_support,
                model.x_hat: x_query, model.y_hat_idx: y_query })
        
        print('a')              

        
