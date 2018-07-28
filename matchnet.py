import os
import numpy as np
import cv2
import tensorflow as tf
from omniloader import OmniglotLoader as og

class MatchNet():

    eps = 1e-10
    lean_rate = 5e-6
    global_step = tf.Variable(0, trainable=False, name='global_step')
    conv_param = {'k_sz':3, 'f_sz':64, 'c_sz':1, 'n_stack':4}
    n_supports = tf.placeholder(tf.int32)
    n_way = tf.placeholder(tf.int32)

    # batch = one matching task, i.e. support set vs query image
    x_i = tf.placeholder(tf.float32, shape=[None, og.im_size, og.im_size, og.im_channel])
    y_i_idx = tf.placeholder(tf.int32, shape=[None]) # n_support
    x_hat = tf.placeholder(tf.float32, shape=[1, og.im_size, og.im_size, og.im_channel])
    y_hat_idx = tf.placeholder(tf.int32, shape=[None]) # batch size
    
    y_i = tf.one_hot(y_i_idx, n_way)
    y_hat = tf.one_hot(y_hat_idx, n_way)

    def convnet_encoder(self, inputs, reuse=False):
        k_sz = self.conv_param['k_sz']
        f_sz = self.conv_param['f_sz']
        c_sz = self.conv_param['c_sz']
        n_stack = self.conv_param['n_stack']
        layer = inputs
        for l in range(n_stack):
            with tf.variable_scope('conv_{}'.format(l)):
                filters = tf.get_variable('filter', [k_sz, k_sz, c_sz, f_sz])
                beta = tf.get_variable('BN_beta', [f_sz], initializer=tf.constant_initializer(0.0))
                gamma = tf.get_variable('BN_gamma', [f_sz], initializer=tf.constant_initializer(1.0))
                c_sz = f_sz
                Z = tf.nn.conv2d(layer, filters, strides=[1,1,1,1], padding='SAME')
                mu, sig = tf.nn.moments(Z, [0,1,2])
                Z_tild = tf.nn.batch_normalization(Z, mu, sig, beta, gamma, self.eps)
                activ = tf.nn.relu(Z_tild)
                layer = tf.nn.max_pool(activ, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        return tf.squeeze(layer, [1,2])
    
    def convnet_encoder_No_BN(self, inputs, reuse=False):
        k_sz = self.conv_param['k_sz']
        f_sz = self.conv_param['f_sz']
        c_sz = self.conv_param['c_sz']
        n_stack = self.conv_param['n_stack']
        layer = inputs
        for l in range(n_stack):
            with tf.variable_scope('conv_{}'.format(l)):
                filters = tf.get_variable('filter', [k_sz, k_sz, c_sz, f_sz])
                bias = tf.get_variable('bias', [f_sz], initializer=tf.constant_initializer(0.0))
                c_sz = f_sz
                Z = tf.nn.conv2d(layer, filters, strides=[1,1,1,1], padding='SAME')
                Z_b = tf.nn.bias_add(Z, bias)
                activ = tf.nn.relu(Z_b)
                layer = tf.nn.max_pool(activ, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        return tf.squeeze(layer, [1,2])

    def __init__(self, bn_layer=True, share_encoder=True):
        self.sharing = share_encoder
        scope = 'image_encoder'
        with tf.variable_scope(scope):
            if bn_layer: self.x_hat_encoded = self.convnet_encoder(self.x_hat)
            else: self.x_hat_encoded = self.convnet_encoder_No_BN(self.x_hat)
        
        if not self.sharing:
            scope = 'support_set_encoder'

        with tf.variable_scope(scope, reuse=self.sharing):
            if bn_layer: self.x_i_encoded = self.convnet_encoder(self.x_i)
            else: self.x_i_encoded = self.convnet_encoder_No_BN(self.x_i)
        
        # self.batchsz = tf.shape(self.x_i_encoded)[0]
        self.dotted = tf.squeeze(tf.matmul(tf.tile(tf.expand_dims(self.x_hat_encoded, 0), [self.n_supports,1,1]), 
                        tf.expand_dims(self.x_i_encoded, 2)), [1,2])
        self.x_h2_inv = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(self.x_hat_encoded)), self.eps, float('inf')))
        self.x_i2_inv = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(self.x_i_encoded), 1), self.eps, float('inf')))
        self.cos_sim = tf.multiply(self.dotted, tf.scalar_mul(self.x_h2_inv, self.x_i2_inv))
        self.attention = tf.nn.softmax(self.cos_sim)
        self.prob = tf.matmul(tf.expand_dims(self.attention, 0), self.y_i)
        self.top_1 = tf.nn.in_top_k(self.prob, self.y_hat_idx, 1)
        self.loss = -1*tf.reduce_sum(tf.log(tf.clip_by_value(self.prob, self.eps, 1.0))*self.y_hat)
        optim = tf.train.AdamOptimizer(learning_rate=self.lean_rate)
        grad = optim.compute_gradients(self.loss)
        self.train_step = optim.apply_gradients(grad)

if __name__ == '__main__':
    
    loader = og(0)
    model = MatchNet()    
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    print(session.run(tf.report_uninitialized_variables()))

    step, acc_batch = 0, 100
    acc_train, acc_loss, acc_test = [], [], []
    while True:
        N_way = 5 # np.random.choice(np.arange(5,11))
        k_shot = 1 # np.random.choice(5)+1
        x_support, y_support, x_query, y_query = loader.getTrainSample_NoBatch(N_way, k_shot)

        [ _, loss_, top1, x_h_, x_i_, dot_, x_hi, x_ii, sim, prob, y_i_, atten, y_hat_ ] = session.run([model.train_step,
                model.loss, model.top_1, model.x_hat_encoded, 
                model.x_i_encoded, model.dotted, model.x_h2_inv, model.x_i2_inv,
                model.cos_sim, model.prob, model.y_i, model.attention, model.y_hat], feed_dict={
                model.n_supports: N_way*k_shot, model.n_way: N_way,
                model.x_i: x_support, model.y_i_idx: y_support,
                model.x_hat: x_query, model.y_hat_idx: y_query })
        
        _, _, n_epoch = loader.getStatus()
        # print('{}({}): {}, {}'.format(step, n_epoch, top1, loss_))
        
        if n_epoch % 10 == 0 and n_epoch != 0:
            N_way_test = 5 # np.random.choice(10)+1
            k_shot_test = 1 # np.random.choice(5)+1
            x_support_test, y_support_test, x_query_test, y_query_test, origin_i, origin_hat = loader.getTestSample_NoBatch(N_way_test, k_shot_test)
            [prob_t, top_1_t] = session.run([model.prob, model.top_1], feed_dict={
                model.n_supports: N_way_test*k_shot_test, model.n_way: N_way_test,
                model.x_i: x_support_test, model.y_i_idx: y_support_test,
                model.x_hat: x_query_test, model.y_hat_idx: y_query_test })
            
            acc_test.append(top_1_t[0])
            if len(acc_test) == acc_batch:
                acc_tmp = np.array(acc_test)*1
                acc_m = np.sum(acc_tmp)/acc_batch
                print('\ttest accuracy: {:2.2%}'.format(acc_m))
                acc_test.clear()
                
            # print('\ttest acc: {}'.format(top_1_t))
            if top_1_t[0]: # sanity check code
                max_loc = np.argmax(prob_t[0])
                clss = [origin_i[chk] for chk in origin_i if origin_i[chk][0] == max_loc]
                for c_j in clss:
                    if c_j[1] != clss[0][1] or c_j[1] != origin_hat[1]:
                        min_loc = np.argmax(prob_t[0])
                        if np.fabs(prob_t[0][max_loc]-prob_t[0][min_loc]) > model.eps:
                            print('error 1')
        
        acc_train.append(top1[0])
        acc_loss.append(loss_)

        if step % acc_batch == 0 and step != 0:
            num_s = len(acc_loss)
            loss_m = sum(acc_loss)/num_s
            acc_temp = np.array(acc_train)*1
            acc_m = np.sum(acc_temp)/num_s
            print('{}({}): {:2.2%}, {}'.format(step, n_epoch, acc_m, loss_m))
            acc_train.clear()
            acc_loss.clear()
        step += 1
        

        
