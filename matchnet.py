import os
import numpy as np
import cv2
import tensorflow as tf
from omniloader import OmniglotLoader as og

class attLSTM(tf.contrib.rnn.RNNCell):
    def __init__(self, num_hidd, g_s, K):
        self.num_hid = num_hidd
        self.lstmcell = tf.contrib.rnn.LSTMCell(num_hidd,
            initializer=tf.contrib.layers.xavier_initializer())
        self.g_s = g_s # n_support * filter_sz
        self.K = K
        self.g_mag_inv = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(self.g_s), 1), 1e-10, float('inf')))

    @property
    def state_size(self):
        return self.lstmcell.state_size
    
    @property
    def output_size(self):
        return self.lstmcell.output_size
    
    def __call__(self, inputs, state):
        h_k = state.h + inputs
        logits = tf.expand_dims(tf.multiply(tf.squeeze(tf.matmul(tf.expand_dims(self.g_s, 1), tf.tile(tf.expand_dims(h_k, 2), [self.K, 1, 1]))), self.g_mag_inv), 0) # norm by mag of g_encode
        att = tf.transpose(tf.nn.softmax(logits)) # n_support * 1
        r_k = tf.reduce_sum(tf.multiply(self.g_s, att), 0, keepdims=True)
        h_concat = tf.reshape( tf.concat([h_k, r_k], 1), [1, 2*self.num_hid] )
        h_in = tf.layers.dense(inputs=h_concat, units=self.num_hid, use_bias=False,
            kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.tanh, name='fce_concat_reduce')
        
        h_hat, new_state = self.lstmcell(inputs, tf.contrib.rnn.LSTMStateTuple(c=state.c, h=h_in))
        return h_hat, new_state


class MatchNet():

    eps = 1e-10  
    lean_rate = 1e-5
    global_step = tf.Variable(0, trainable=False, name='global_step')
    conv_param = {'k_sz':3, 'f_sz':64, 'c_sz':1, 'n_stack':4}
    n_supports = tf.placeholder(tf.int32)
    n_way = tf.placeholder(tf.int32)

    # batch = one matching task, i.e. support set vs query image
    x_i = tf.placeholder(tf.float32, shape=[None, og.im_size, og.im_size, og.im_channel])
    y_i_idx = tf.placeholder(tf.int32, shape=[None]) # n_support
    x_hat = tf.placeholder(tf.float32, shape=[1, og.im_size, og.im_size, og.im_channel])
    y_hat_idx = tf.placeholder(tf.int32, shape=[1]) # batch size
    
    y_i = tf.one_hot(y_i_idx, n_way)
    y_hat = tf.one_hot(y_hat_idx, n_way)

    def lenet_encoder(self, inputs, reusing=False):
        with tf.variable_scope('lenet'):
            filter1 = tf.get_variable('conv1', [5,5,1,6])
            filter2 = tf.get_variable('conv2', [5,5,6,16])
            layers = inputs
            layers = tf.nn.conv2d(layers, filter1, strides=[1,1,1,1], padding='SAME')
            layers = tf.nn.relu(layers)
            layers = tf.nn.max_pool(layers, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
            
            layers = tf.nn.conv2d(layers, filter2, strides=[1,1,1,1], padding='VALID')
            layers = tf.nn.relu(layers)
            layers = tf.nn.max_pool(layers, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
            layers = tf.reshape(layers, [-1, 16*5*5])
            
            layers = tf.layers.dense(inputs=layers, units=120, activation=tf.nn.relu)
            layers = tf.layers.dense(inputs=layers, units=84, activation=tf.nn.relu)
            layers = tf.layers.dense(inputs=layers, units=64)
        return layers

    def convnet_encoder(self, inputs, reusing=False):
        k_sz = self.conv_param['k_sz']
        f_sz = self.conv_param['f_sz']
        c_sz = self.conv_param['c_sz']
        n_stack = self.conv_param['n_stack']
        layer = inputs
        for l in range(n_stack):
            with tf.variable_scope('conv_{}'.format(l)):
                filters = tf.get_variable('filter', [k_sz, k_sz, c_sz, f_sz], 
                    initializer=tf.contrib.layers.xavier_initializer())
                beta = tf.get_variable('BN_beta', [f_sz], initializer=tf.constant_initializer(0.0))
                gamma = tf.get_variable('BN_gamma', [f_sz], initializer=tf.constant_initializer(1.0))
                c_sz = f_sz
                Z = tf.nn.conv2d(layer, filters, strides=[1,1,1,1], padding='SAME')
                mu, sig = tf.nn.moments(Z, [0,1,2])
                Z_tild = tf.nn.batch_normalization(Z, mu, sig, beta, gamma, self.eps)
                activ = tf.nn.relu(Z_tild)
                layer = tf.nn.max_pool(activ, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        return tf.squeeze(layer, [1,2])
    
    def convnet_encoder_No_BN(self, inputs, reusing=False):
        k_sz = self.conv_param['k_sz']
        f_sz = self.conv_param['f_sz']
        c_sz = self.conv_param['c_sz']
        n_stack = self.conv_param['n_stack']
        layer = inputs
        for l in range(n_stack):
            with tf.variable_scope('conv_{}'.format(l)):
                filters = tf.get_variable('filter', [k_sz, k_sz, c_sz, f_sz],
                    initializer=tf.contrib.layers.xavier_initializer())
                bias = tf.get_variable('bias', [f_sz], initializer=tf.constant_initializer(0.0))
                c_sz = f_sz
                Z = tf.nn.conv2d(layer, filters, strides=[1,1,1,1], padding='SAME')
                Z_b = tf.nn.bias_add(Z, bias)
                activ = tf.nn.relu(Z_b)
                layer = tf.nn.max_pool(activ, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        return tf.squeeze(layer, [1,2])
    
    def fce_support_set(self, inputs):
        with tf.variable_scope('fce_g'):
            hidd_size = self.conv_param['f_sz']
            g_f = tf.expand_dims(inputs, 0)
            lstm_f = tf.contrib.rnn.LSTMCell(hidd_size, name='lstm_f')
            init_f = lstm_f.zero_state(1, dtype=tf.float32)
            h_f, _ = tf.nn.dynamic_rnn(lstm_f, g_f, sequence_length=self.n_supports, initial_state=init_f)

            g_b = tf.expand_dims(tf.reverse(inputs, [0]), 0)
            lstm_b = tf.contrib.rnn.LSTMCell(hidd_size, name='lstm_b')
            init_b = lstm_b.zero_state(1, dtype=tf.float32)
            h_b, _ = tf.nn.dynamic_rnn(lstm_b, g_b, sequence_length=self.n_supports, initial_state=init_b)
            h_b = tf.reverse(h_b, [1])
            
            g_new = tf.squeeze(h_f, [0]) + tf.squeeze(h_b, [0]) + inputs
            return g_new
    
    def fce_query_image(self, inputs, g_s):
        with tf.variable_scope('fce_f'):
            hidd_size = self.conv_param['f_sz']
            attlstm = attLSTM(hidd_size, g_s, self.n_supports)
            init_st = attlstm.lstmcell.zero_state(1, dtype=tf.float32)
            f_x = tf.tile(tf.expand_dims(inputs, 1), [1, self.n_supports, 1])
            h_hat, att_state = tf.nn.dynamic_rnn(attlstm, f_x, 
                sequence_length=self.n_supports, initial_state=init_st)
            
            return h_hat[:,-1,:] + inputs

    def __init__(self, bn_layer=True, share_encoder=True, fce=False):
        self.sharing = share_encoder
        scope = 'image_encoder'
        with tf.variable_scope(scope):
            if bn_layer == True: 
                self.x_hat_encoded = self.convnet_encoder(self.x_hat)
            elif bn_layer == False: 
                self.x_hat_encoded = self.convnet_encoder_No_BN(self.x_hat)
            elif bn_layer == None:
                self.x_hat_encoded = self.lenet_encoder(self.x_hat)
        
        if not self.sharing:
            scope = 'support_set_encoder'

        with tf.variable_scope(scope, reuse=self.sharing):
            if bn_layer == True: 
                self.x_i_encoded = self.convnet_encoder(self.x_i, self.sharing)
            elif bn_layer == False: 
                self.x_i_encoded = self.convnet_encoder_No_BN(self.x_i, self.sharing)
            elif bn_layer == None:
                self.x_i_encoded = self.lenet_encoder(self.x_i, self.sharing)
            
        
        # self.batchsz = tf.shape(self.x_i_encoded)[0]
        if fce:
            self.x_i_encoded = self.fce_support_set(self.x_i_encoded)
            self.x_hat_encoded = self.fce_query_image(self.x_hat_encoded, self.x_i_encoded)

        self.tiled = tf.tile(tf.expand_dims(self.x_hat_encoded, 0), [self.n_supports,1,1])
        self.dotted = tf.squeeze(tf.matmul(self.tiled,tf.expand_dims(self.x_i_encoded, 2)), [1,2])
        self.x_i2_inv = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(self.x_i_encoded), 1), self.eps, float('inf')))
        self.cos_sim = tf.multiply(self.dotted, self.x_i2_inv)
        # self.x_h2_inv = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(self.x_hat_encoded)), self.eps, float('inf'))) 
        # self.cos_sim = tf.multiply(self.dotted, tf.scalar_mul(self.x_h2_inv, self.x_i2_inv)) # For the stability of BP
        self.attention = tf.nn.softmax(self.cos_sim)
        self.prob = tf.matmul(tf.expand_dims(self.attention, 0), self.y_i)
        self.top_1 = tf.nn.in_top_k(self.prob, self.y_hat_idx, 1)
        self.loss = -1*tf.reduce_sum(tf.log(tf.clip_by_value(self.prob, self.eps, 1.0))*self.y_hat)
        optim = tf.train.AdamOptimizer(learning_rate=self.lean_rate)
        grad = optim.compute_gradients(self.loss)
        self.train_step = optim.apply_gradients(grad)
        tf.summary.scalar('loss', self.loss)

if __name__ == '__main__':
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    loader = og(0)
    model = MatchNet(bn_layer=False, fce=True)    

    save_name = 'model/matchnet.ckpt'
    saver = tf.train.Saver()
    session = tf.Session()
    if tf.train.checkpoint_exists(save_name):
        #saver.restore(session, save_name)
        session.run(tf.global_variables_initializer())
        print('model loaded...')
    else:
        session.run(tf.global_variables_initializer())
        print('model initialized...')

    step, acc_batch = 0, 100
    acc_train, acc_loss, acc_test = [], [], []
    warn_tie, warn_uniform = 0, 0
    min_loss = float('inf')
    
    merged = tf.summary.merge_all()
    log_writer = tf.summary.FileWriter('logs', session.graph)
    while True:
        N_way = np.random.choice(np.arange(5,11))
        k_shot = np.random.choice(5)+1
        x_support, y_support, x_query, y_query = loader.getTrainSample_NoBatch(N_way, k_shot, False)
        # x_support, y_support, x_query, y_query = loader.getFakeSample(N_way, k_shot, False)
        # x_support = np.squeeze(x_support, axis=0)
        # y_support = np.squeeze(y_support, axis=0)

        [ _, summary,loss_, top1, x_h_, x_h0, x_i_, dot_, x_ii, sim, prob, y_i_, atten, y_hat_, tiled ] = session.run([model.train_step, merged,
                model.loss, model.top_1, model.x_hat_encoded, model.x_hat,
                model.x_i_encoded, model.dotted, model.x_i2_inv,
                model.cos_sim, model.prob, model.y_i, model.attention, model.y_hat, model.tiled], feed_dict={
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
                        print('\twarning: tie...')
                        warn_tie += 1
                min_loc = np.argmin(prob_t[0])
                if np.fabs(prob_t[0][max_loc]-prob_t[0][min_loc]) <= model.eps:
                    print('\tuniformly distributed')
                    warn_uniform += 1
        
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
            log_writer.add_summary(summary, global_step=step)
            log_writer.flush()
            if n_epoch % 5 and n_epoch != 0 and min_loss > loss_m:
                min_loss = loss_m
                saver.save(session,save_name, global_step=step)

        step += 1