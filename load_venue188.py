##
## author: zjl
## time:2017/5/6

import numpy as np
import tables
import random

class venue_data(object):
    ## data preparation for model training, validating, and testing
    def __init__(self,file_name):

        self.name = file_name

        self.f = tables.open_file(self.name, 'r')

        self.X_train = self.f.root.X_train
        self.y_train = self.f.root.y_train

        self.X_valid = self.f.root.X_valid
        self.y_valid = self.f.root.y_valid

        self.X_test = self.f.root.X_test
        self.y_test = self.f.root.y_test



    def split_feature(self, x, modality_flag, modality_index):
        '''
        get the modality feature
        :param x: feature set
        :param modality_flag: 0,1,and 2 indicates visual,acoustic,and textual feature respectively
        :param modality_index: feature indices for each modality
        :return: intention modality feature
        '''
        #

        if modality_flag == 0:
            return x[:,modality_index[0]:modality_index[1]]

        elif modality_flag == 1:
            return x[:, modality_index[1]:modality_index[2]]

        else:
            return x[:, modality_index[2]:modality_index[3]]


    def split(self,x,modality_index):
        # split the data into multi-views
        '''
        :param x: feature set
        :param modality_index: feature indices for each modality
        :return: modalities set
        '''
        x_v = x[:,modality_index[0]:modality_index[1]]
        x_a = x[:, modality_index[1]:modality_index[2]]
        x_t = x[:, modality_index[2]:modality_index[3]]
        return x_v,x_a,x_t

    def get_train_ids(self):
        # get the training videos ids
        return self.f.root.id_train

    def get_valid_ids(self):
        # get the validation videos ids
        return self.f.root.id_valid

    def get_test_ids(self):
        # get the testing videos ids
        return self.f.root.id_test

    def get_train_lables(self):
        # get the training data label (from 0)
        return self.f.root.y_train

    def get_valid_lables(self):

        return self.f.root.y_valid

    def get_test_lables(self):

        return self.f.root.y_test

    def get_train_num(self):
        # get the number of training data
        return self.y_train.shape[0]

    def get_valid_num(self):
        # get the number of valid data
        return self.y_valid.shape[0]

    def get_test_num(self):
        # get the number of testing data
        return self.y_test.shape[0]

    def next_batch(self,batch_size,i):
        # fetch the batch data for training
        # batch_size: the size of each batch
        # i:the index of batch
        # output:
        # X: features
        # y: labels
        X = self.X_train[batch_size * i: batch_size * (i+1)]
        y = self.y_train[batch_size * i: batch_size * (i+1)]

        return X,y

    def one_hot_next_batch(self,batch_size,i):
        # convert the label to one-hot representation
        # input:
        # batch_size: the size of each batch
        # i:the index of batch
        # output:
        # X: features
        # y: one hot representation of labels

        X = self.X_train[batch_size * i: batch_size * (i+1)]
        y = self.y_train[batch_size * i: batch_size * (i+1)]

        y = self.one_hot(y)

        return X,y

    def one_hot(self,y):
        # one hot representation in venue188 dataset
        # input:
        # y: label
        # output: one hot
        depth = 188
        num = y.shape[0]
        temp = np.zeros((num,depth),dtype=np.float32)
        for i in range(188):
            index = np.argwhere(y==i)
            temp[index,i] = 1.0

        return temp

    def train_random_sample(self,batch_size):
        # randomly sample from training data
        # input:
        # batch_size: the size of each batch
        # i:the index of batch
        # output:
        # sample_index:sample index of data
        # X: features
        # y: labels
        self.get_train_ids()
        sample_index = random.sample(self.train_ids, batch_size)

        X = self.X_train[sample_index,:]
        y = self.y_train[sample_index]

        return sample_index, X,y

    def close(self):
        # close the hdf5 file
        self.f.close()
