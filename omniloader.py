
import os
import cv2
import numpy as np

class OmniglotLoader():
    
    dataloc = 'omniglot/'
    backset = dataloc + 'images_background/'
    evalset = dataloc + 'images_evaluation/'
    
    n_examples_per_char = 20
    langs1 = os.listdir(backset)
    langs2 = os.listdir(evalset)
    
    train_chars = {}
    test_chars = {}
    for lang in langs1:
        for char in os.listdir(backset+lang):
            files = os.listdir(backset+lang+'/'+char)
            class_id = int(files[0].split('_')[0])
            if class_id <= 1200:
                train_chars[class_id] = backset+lang+'/'+char
            else:
                test_chars[class_id] = backset+lang+'/'+char

    for lang in langs2:
        for char in os.listdir(evalset+lang):
            files = os.listdir(evalset+lang+'/'+char)
            class_id = int(files[0].split('_')[0])
            if class_id <= 1200:
                train_chars[class_id] = evalset+lang+'/'+char
            else:
                test_chars[class_id] = evalset+lang+'/'+char
    
    n_class_train = len(train_chars) 
    n_class_test = len(test_chars)
    im_size = 28
    im_channel = 1
    
    def __init__(self, rdseed):
        np.random.seed(rdseed)
        self.train_ptr = 0
        self.epoch = 0
        self.permute = np.random.permutation(self.n_class_train) + 1

    def shuffling(self):
        np.random.shuffle(self.permute)
        self.epoch += 1
        self.train_ptr = 0
        
    def getFakeSample(self, N_way, k_shot, disp=False):
        batch_size = 1
        n_support = N_way*k_shot
        x_i_support = np.zeros([batch_size, n_support, self.im_size, self.im_size, self.im_channel]) # one channel (the last dimension)
        y_i_support = np.zeros([batch_size, n_support])
        x_hat = np.zeros([batch_size, self.im_size, self.im_size, self.im_channel])
        y_hat = np.zeros([batch_size])
        clss = np.random.choice(N_way)
        sample_perm = np.random.permutation(n_support)
        sample_ind = 0
        for s in range(N_way):
            x_i_support[0,sample_perm[sample_ind],:,:,:] = np.ones([1, self.im_size, self.im_size, self.im_channel])/(3*(s+1))
            y_i_support[0,sample_perm[sample_ind]] = s
            sample_ind += 1
        x_hat[0] = np.ones([self.im_size, self.im_size, self.im_channel])/(3*(clss+1))
        y_hat[0] = clss
        return x_i_support, y_i_support, x_hat, y_hat


    def getTrainSample(self, batch_size, N_way, k_shot, disp=False):
        n_support = N_way*k_shot
        x_i_support = np.zeros([batch_size, n_support, self.im_size, self.im_size, self.im_channel]) # one channel (the last dimension)
        y_i_support = np.zeros([batch_size, n_support])
        x_hat = np.zeros([batch_size, self.im_size, self.im_size, self.im_channel])
        y_hat = np.zeros([batch_size])

        for b in range(batch_size):
            need_shuffle = True if (self.train_ptr + N_way >= self.n_class_train) else False                

            perm_idx = np.arange(self.train_ptr, self.train_ptr + N_way)%self.n_class_train
            self.train_ptr = (self.train_ptr + N_way)

            label_set = [ [self.permute[i] , self.train_chars[self.permute[i]]] for i in perm_idx ]            
            sample_perm = np.random.permutation(n_support)
            sample_ind = 0
            hat_class = np.random.choice(N_way)
            for c, clss in enumerate(label_set):
                ex_ind = np.random.choice(self.n_examples_per_char, k_shot, replace=False) + 1
                for ex in ex_ind:
                    sample_name = clss[1]+'/{:04d}_{:02d}.png'.format(clss[0], ex)
                    img = cv2.imread(sample_name, -1)
                    img = cv2.resize(img, (self.im_size,self.im_size), interpolation=cv2.INTER_AREA)
                    x_i_support[b, sample_perm[sample_ind], :, :, 0] = img.astype(np.float)/127.5 - 1
                    y_i_support[b, sample_perm[sample_ind]] = c
                    sample_ind += 1
                if c == hat_class:
                    allsmps = np.arange(self.n_examples_per_char)+1
                    remains = np.setdiff1d(allsmps, ex_ind)
                    sample_name = clss[1]+'/{:04d}_{:02d}.png'.format(clss[0], np.random.choice(remains))
                    img = cv2.imread(sample_name, -1)
                    img = cv2.resize(img, (self.im_size, self.im_size), interpolation=cv2.INTER_AREA)
                    x_hat[b, :, :, 0] = img.astype(np.float)/127.5-1
                    y_hat[b] = c
            if need_shuffle: self.shuffling()
        if disp: self.displayImage(x_i_support, x_hat)
        return np.float32(x_i_support), y_i_support, np.float32(x_hat), y_hat
    
    def getTrainSample_NoBatch(self, N_way, k_shot, disp=False):
        x_i, y_i, x_h, y_h = self.getTrainSample(1, N_way, k_shot, disp)
        x_i_ = np.squeeze(x_i, axis=0)
        y_i_ = np.squeeze(y_i, axis=0)
        return x_i_, y_i_, x_h, y_h

    def getTestSample(self, batch_size, N_way, k_shot):
        n_support = N_way*k_shot
        x_i_support = np.zeros([batch_size, n_support, self.im_size, self.im_size, self.im_channel])
        y_i_support = np.zeros([batch_size, n_support])
        x_hat = np.zeros([batch_size, self.im_size, self.im_size, self.im_channel])
        y_hat = np.zeros([batch_size])
        origins_i, origins_hat = [], []
        for b in range(batch_size):
            chosen = np.random.choice(list(self.test_chars.keys()), N_way, False)
            label_set = [ [l, self.test_chars[l]] for l in chosen ]
            sample_perm = np.random.permutation(n_support)
            sample_ind = 0
            hat_class = np.random.choice(N_way)
            or_i = {}
            for c, clss in enumerate(label_set):
                ex_ind = np.random.choice(self.n_examples_per_char, k_shot, replace=False) + 1
                for ex in ex_ind:
                    sample_name = clss[1]+'/{:04d}_{:02d}.png'.format(clss[0], ex)
                    img = cv2.imread(sample_name, -1)
                    img = cv2.resize(img, (self.im_size, self.im_size), interpolation=cv2.INTER_AREA)
                    x_i_support[b, sample_perm[sample_ind],:,:,0] = img.astype(np.float)/127.5 - 1
                    y_i_support[b, sample_perm[sample_ind]] = c
                    or_i[sample_perm[sample_ind]] = [c, clss[0], sample_name]
                    sample_ind += 1
                if c == hat_class:
                    allsmps = np.arange(self.n_examples_per_char)+1
                    remains = np.setdiff1d(allsmps, ex_ind)
                    sample_name = clss[1]+'/{:04d}_{:02d}.png'.format(clss[0], np.random.choice(remains))
                    img = cv2.imread(sample_name, -1)
                    img = cv2.resize(img, (self.im_size, self.im_size), interpolation=cv2.INTER_AREA)
                    x_hat[b, :, :, 0] = img.astype(np.float)/127.5 - 1
                    y_hat[b] = c
                    origins_hat.append([c, clss[0], sample_name])
            origins_i.append(or_i)
        return np.float32(x_i_support), y_i_support, np.float32(x_hat), y_hat, origins_i, origins_hat
    
    def getTestSample_NoBatch(self, N_way, k_shot):
        x_i, y_i, x_h, y_h, o_i, o_h = self.getTestSample(1, N_way, k_shot)
        x_i_ = np.squeeze(x_i, axis=0)
        y_i_ = np.squeeze(y_i, axis=0)
        return x_i_, y_i_, x_h, y_h, o_i[0], o_h[0]

    def getStatus(self):
        return self.train_ptr, self.n_class_train, self.epoch

    def displayImage(self, x_i, x_h):
        sz = x_i.shape[1]
        for s in range(sz):
            cv2.namedWindow('x_{}'.format(s), cv2.WINDOW_NORMAL)
            cv2.imshow('x_{}'.format(s), x_i[0,s,:,:,:])
        cv2.namedWindow('x_h', cv2.WINDOW_NORMAL)
        cv2.imshow('x_h', x_h[0,:,:,:])
        cv2.waitKey()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    loader = OmniglotLoader(0)
    batch_size = 1
    N_way, k_shot = 5, 1
    while True:
        #N_way = np.random.choice(10)+1
        #k_shot = np.random.choice(5)+1
        x_i, y_i, x_hat, y_hat = loader.getTrainSample(batch_size, N_way, k_shot)
        # x_i, y_i, x_hat, y_hat, ori_i, ori_hat = loader.getTestSample(batch_size, N_way, k_shot)
        for b in range(batch_size):
            im_s = x_i[b, np.where(y_i[b] == y_hat[b]), : ,:, :]
            n_s = im_s.shape[1]
            for n in range(n_s):
                cv2.namedWindow('s{}'.format(n), cv2.WINDOW_NORMAL)
                cv2.imshow('s{}'.format(n), im_s[0,n,:,:, :])
            im_h = x_hat[b,:,:,:]
            cv2.namedWindow('h', cv2.WINDOW_NORMAL)
            cv2.imshow('h', im_h)
            print(N_way, k_shot, loader.getStatus()) 
            cv2.waitKey()       
            
        
    print(loader.train_chars)
    print(loader.test_chars)
    