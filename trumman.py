'''
Created on Mar 1, 2016

@author: zjl
'''

import numpy as np
import _pickle as pickle
# import preprocess
import timeit
import tables

import theano
import theano.tensor as T

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.cross_validation import StratifiedShuffleSplit

import matplotlib.pyplot as plt


def split_view(X, view_index):
    view_num = view_index.__len__() - 1
    T = []
    for i in range(view_num):
        start = view_index[i]
        end_point = view_index[i + 1]
        x = X[:, start:end_point]
        T.append(x)

    return T

def split_train_valid_test_set(feature, labels):
    # generate subsamples of any size while retaining the structure of the whole dataset
    # train, test = cross_validation.train_test_split(videos, test_size = 0.3, random_state=0)
    ss = StratifiedShuffleSplit(labels, n_iter=1, test_size=0.3, random_state=0)

    for train_index, test_index in ss:
        y_train_temp, y_test = labels[train_index], labels[test_index]
        X_train_temp, X_test = feature[train_index], feature[test_index]

    sss = StratifiedShuffleSplit(y_train_temp, n_iter=1, test_size=0.3, random_state=1)

    for train_index, valid_index in sss:
        X_train, X_valid = X_train_temp[train_index], X_train_temp[valid_index]
        y_train, y_valid = y_train_temp[train_index], y_train_temp[valid_index]

    train_set = X_train, y_train
    valid_set = X_valid, y_valid
    test_set = X_test, y_test

    dataset = [train_set, valid_set, test_set]

    return dataset

def shared_dataset(data_xy, borrow=True):

    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)

    return shared_x, T.cast(shared_y, 'int32')

def load_data_file(file, share=True):
    f = tables.openFile(file, 'r')

    root = f.root
    features = root.features
    labels = root.labels

    #     features = np.matrix(features,dtype=np.float16)
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    # normalization
    # features = min_max_scaler(features)

    # split into train valid and test data
    dataset = split_train_valid_test_set(features, labels)

    train_set, valid_set, test_set = dataset
    if share == True:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]

        return rval
    else:
        return dataset

def plot_w(W):
    data = np.array(W)

    fig = plt.figure()
    plt.clf()
    res = plt.imshow(data, interpolation='nearest')
    plt.show()
    plt.xticks(np.arange(0, 10))
    plt.yticks(np.arange(0, 10))

    cb = fig.colorbar(res)


def getA(X, B, l1, l2):
    '''
    \mathbf{A}^s = (\lambda_1 \mathbf{X}^{sT} \mathbf{X}^s +    \\
             \lambda_2 \mathbf{I})^{-1} (\lambda_1 \mathbf{X}^{sT} \mathbf{B})

    :return A, mapping matrix
    :type, numpy array
    '''
    view_num = X.__len__()
    A = []
    for i in range(view_num):
        view_dim = X[i].shape[1]
        I = np.eye(view_dim, dtype=np.float32)

        #         inv_part = l1 * (np.transpose(X[i]) * X[i]) + l2 * I
        #         part = l1 * np.transpose(X[i]) * B
        #         start_time = timeit.default_timer()
        inv_part = l1 * np.dot(X[i].T, X[i]) + l2 * I
        #         end_time = timeit.default_timer()
        #         print('time is %f' %(end_time - start_time))

        part = l1 * np.dot(X[i].T, B)
        #         inv_part = l1 * X[i].transpose() * X[i] + l2 * I
        #         part = l1 * X[i].transpose() * B
        #         inv_part = l1 * np.dot(X[i].transpose(), X[i])+ l2 * I
        #         part = l1 * np.dot(X[i].transpose(), B)

        A_i = np.dot(np.linalg.inv(inv_part), part)
        A.append(A_i)

    return A


def getB(X, Y, W, A, l1):
    '''
    \mathbf{B} = ( \mathbf{Y} \mathbf{W}^T + \lambda_1 \sum_{s=1}^S \mathbf{X}^s \mathbf{A}^s)  \\
                                (\lambda_1 S \mathbf{I} + \mathbf{W} \mathbf{W}^T)^{-1}

    :return: B, common space respective
    :type: numpy array
    '''
    latent_num, task_num = W.shape

    view_num = X.__len__()

    represent = 0

    for i in range(view_num):
        represent += np.dot(X[i], A[i])

    first_term = np.dot(Y, W.T)

    Idensity_matrix = np.eye(latent_num)

    inv_part = l1 * np.dot(view_num, Idensity_matrix) + np.dot(W, W.T)

    B = np.dot((first_term + l1 * represent), np.linalg.inv(inv_part))

    return B


def getD(W, ev, Gv):
    '''
    {D^{t}}_{kk} = \sum_{ \{v\in\mathcal{V},| t \in v \}} \frac{ e_v^2}{q_{k,v}}

    :return: D
    :type: list of array
    '''
    latent_num, task_num = W.shape

    nodes_num = ev.__len__()

    D_list = []
    q = []
    for t in range(task_num):
        q_total = 0
        D = np.zeros((latent_num, latent_num), dtype=np.float32)
        Q = np.zeros((latent_num, nodes_num), dtype=np.float32)

        for v in range(nodes_num):
            w_v = W[:, Gv[v]]
            if t in Gv[v]:
                for k in range(latent_num):
                    Q[k, v] = ev[v] * np.linalg.norm(w_v[k], 2)
                    q_total += Q[k, v]

        if q_total != 0:
            Q = Q / q_total

        for k in range(latent_num):
            for v in range(nodes_num):
                if Q[k, v] != 0:
                    D[k, k] += (ev[v] * ev[v]) / Q[k, v]

        D_list.append(D)
        q.append(q_total)

    q = np.sum(q)
    return D_list, q


def getW(B, Y, Q, l3):
    '''
    \mathbf{w_{t}} = (\mathbf{B}^T \mathbf{B} + \\
            \lambda_3 \mathbf{Q}^{t})^{-1} \mathbf{B}^T \mathbf{y_{t}}
    :return, W, weights
    :type, list
    '''
    task = Y.shape[1]
    k_dim = B.shape[1]
    W = []
    for i in range(task):
        inv_part = np.dot(B.T, B) + l3 * Q[i]
        inv_part = np.linalg.inv(inv_part)

        #         W[i] = np.dot(np.dot(inv_part, B.T),Y[i])

        w = np.dot(inv_part, np.dot(B.T, Y.T[i]))
        W.append(w)

    return W


def get_task_W(X, Y, Q, l1):
    '''
    \mathbf{w_{t}} = (\mathbf{X}^T \mathbf{X} + \\
            \lambda_3 \mathbf{Q}^{t})^{-1} \mathbf{X}^T \mathbf{y_{t}}
    :return, W, weights
    :type, list
    '''
    task = Y.shape[1]
    k_dim = X.shape[1]
    W = []
    for i in range(task):
        inv_part = np.dot(X.T, X) + l1 * Q[i]
        inv_part = np.linalg.inv(inv_part)

        #         W[i] = np.dot(np.dot(inv_part, B.T),Y[i])

        w = np.dot(inv_part, np.dot(X.T, Y.T[i]))
        W.append(w)

    return W


def task_optimize(X, Y, Gv, ev, iteration_num, l1):
    '''
    \min \limits_{\mathbf{W}, \mathbf{Q}} &\frac{1}{2} \sum_{t=1}^T \| \mathbf{y}_t - \\
                \mathbf{X} \mathbf{w}_t \|^2_F +  \\
                \frac{\lambda_3}{2} \sum_{t=1}^T \mathbf{w}_t^T \mathbf{Q}^{t} \mathbf{w}_t
    '''
    DEFAULT_TOLERANCE = 1
    feature_dim = X.shape[1]
    task_dim = Y.shape[1]

    W = np.random.rand(feature_dim, task_dim)

    obj_ex = 0
    for iter in range(iteration_num):
        D, q = getD(W, ev, Gv)
        W = get_task_W(X, Y, D, l1)
        W = np.asarray(W, dtype=np.float32)  # list to array
        W = W.T

        first_term = 0.5 * np.linalg.norm((Y - np.dot(X, W)))
        print('first term loss is %f' % first_term)

        second_term = 0
        for i in range(task_dim):
            second_term += np.dot(np.dot(W.T[i], D[i]), W.T[i])
        second_term = 0.5 * l1 * np.sqrt(second_term)
        print('Second term loss is %f' % second_term)

        obj_value = first_term + second_term
        obj_error = np.abs(obj_value - obj_ex)

        print('%d ===== %f ===== %f' % (iter, obj_value, obj_error))

        if obj_error < DEFAULT_TOLERANCE:
            print('training done')
            break

        obj_ex = obj_value

    return W


def task_prediction(X, W):
    '''
    :param: test_x, test features
    :type: matrix
    :param: W,
    :type: matrix
    :param: A
    :type: matrix

    :purpose: predict the label of test data
    '''
    X_W = np.dot(X, W)

    # get the max
    predict = np.argmax(X_W, axis=1)

    return predict


def task_main():
    Gv = get_tree_group()
    ev = get_group_weight()
    #     ev = ones(ev)

    file = 'leaf_mf_dataset.h5'
    f = tables.openFile(file, 'r')
    # the indices of multi-view feature
    #     view_indices = [0,4096,4296,4396]

    root = f.root
    features = root.features
    labels = root.labels

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    # Provides train/test indices to split data in train test sets.
    sss = StratifiedShuffleSplit(labels, n_iter=5, test_size=0.2, random_state=0)

    macro_p_list = []
    macro_r_list = []
    macro_f_list = []
    micro_f_list = []

    # for each fold
    for train_index, test_index in sss:
        y_train, y_test = labels[train_index], labels[test_index]
        X_train, X_test = features[train_index], features[test_index]

        #         train_set_x = preprocess.split_view(X_train, view_indices)
        #         test_set_x = preprocess.split_view(X_test, view_indices)

        train_num = X_train.shape[0]

        task_indices = np.unique(y_train)
        task_dim = task_indices.__len__()
        if task_dim != np.unique(y_train).__len__():
            print('task dim error')

        Y = np.zeros((train_num, task_dim), dtype=np.int32)
        for i in range(task_dim):
            indices = (y_train == task_indices[i])
            Y[indices, i] = 1

        # parameters
        iteration_num = 10
        l1 = 0.1

        W = task_optimize(X_train, Y, Gv, ev, iteration_num, l1)

        # visualization of W
        #         plot_w(W)

        y_hat = task_prediction(X_test, W)
        y_hat = np.array(y_hat, dtype=np.int32)

        f1_micro = f1_score(y_test, y_hat, average='micro')
        f1_macro = f1_score(y_test, y_hat, average='macro')
        accuracy = accuracy_score(y_test, y_hat)
        cf_matrix = confusion_matrix(y_test, y_hat)
        macro_p = precision_score(y_test, y_hat, average='macro')
        macro_r = recall_score(y_test, y_hat, average='macro')

        macro_p_list.append(macro_p)
        macro_r_list.append(macro_r)
        macro_f_list.append(f1_macro)
        micro_f_list.append(f1_micro)

        #         print('confusion_matrix is : ', cf_matrix)
        print('the micro f1 score of our model is : %f' % f1_micro)
        print('the macro f1 score of our model is : %f, %f, %f' % (macro_p, macro_r, f1_macro))
        print('the accuracy score of our model is : %f' % accuracy)

    print('the average macro precision score of our model is : %f' % np.mean(macro_p_list))
    print('the average macro recall score of our model is : %f' % np.mean(macro_r_list))
    print('the average macro f1 score of our model is : %f' % np.mean(macro_f_list))
    print('the average micro f1 score of our model is : %f' % np.mean(micro_f_list))

    print('the standard deviation macro precision score of our model is : %f' % np.std(macro_p_list))
    print('the standard deviation macro recall score of our model is : %f' % np.std(macro_r_list))
    print('the standard deviation macro f1 score of our model is : %f' % np.std(macro_f_list))
    print('the standard deviation micro f1 score of our model is : %f' % np.std(micro_f_list))


def our_optimize(X, Y, k, Gv, ev, iteration_num, l1, l2, l3):
    '''
    \min\limits_{\mathbf{W},\mathbf{Q}} &\frac{1}{2}   \sum_{t=1}^T \| \mathbf{y}_t -  \\
            \mathbf{B} \mathbf{w}_t \|^2_F +             \\
            \frac{\lambda_1}{2}  \sum _{s=1}^S \| \mathbf{X}^s \mathbf{A}^s - \mathbf{B} \|^2_F +    \\
            &\frac{\lambda_2}{2} \sum_{s=1}^S \| \mathbf{A}^s \|_F^2  +     \\
            \frac{\lambda_3}{2} \sum_{t=1}^T \mathbf{w}_t^T \mathbf{Q}^{t} \mathbf{w}_t
    '''
    DEFAULT_TOLERANCE = 1
    feature_dim = 0
    view_num = X.__len__()

    for i in range(view_num):
        sample_dim, dim_v = X[i].shape
        feature_dim = feature_dim + dim_v

    task_dim = Y.shape[1]

    A = np.random.rand(feature_dim, k)
    B = np.random.rand(sample_dim, k)
    W = np.random.rand(k, task_dim)

    obj_ex = 0
    for iter in range(iteration_num):
        A = getA(X, B, l1, l2)
        B = getB(X, Y, W, A, l1)
        D, q = getD(W, ev, Gv)

        W = getW(B, Y, D, l3)
        #         plot_w(W)

        W = np.asarray(W, dtype=np.float32)  # list to array
        W = W.T
        # object function
        first_term = 0
        third_term = 0
        fourth_term = 0

        for i in range(view_num):
            temp = np.linalg.norm((np.dot(X[i], A[i]) - B))
            first_term += temp

            temp = np.linalg.norm(A[i])
            third_term += temp

        first_term = 0.5 * l1 * first_term
        print('first term loss is %f' % first_term)
        third_term = 0.5 * l2 * third_term
        print('third term loss is %f' % third_term)

        second_term = 0.5 * np.linalg.norm((Y - np.dot(B, W)))
        print('second term loss is %f' % second_term)

        for i in range(task_dim):
            fourth_term += np.dot(np.dot(W.T[i], D[i]), W.T[i])
        fourth_term = 0.5 * l3 * np.sqrt(fourth_term)

        #         fourth_term2 = 0.5 * l3 * q
        #         print(fourth_term1 - fourth_term2)
        print('fourth term loss is %f' % fourth_term)

        # object value
        obj_value = first_term + second_term + third_term + fourth_term

        obj_error = np.abs(obj_value - obj_ex)

        print('%d ===== %f ===== %f' % (iter, obj_value, obj_error))

        if obj_error < DEFAULT_TOLERANCE:
            print('training done')
            break

        obj_ex = obj_value

    return A, B, W


def mtl_class_predict(test_x, W, A):
    '''
    :param: test_x, test features
    :type: matrix
    :param: W,
    :type: matrix
    :param: A
    :type: matrix

    :purpose: predict the label of test data
    '''
    view_num = test_x.__len__()

    sample_num = test_x[0].shape[0]
    latent_dim = A[0].shape[1]

    new_rep = np.zeros((sample_num, latent_dim), dtype=np.float32)

    for i in range(view_num):
        new_rep += np.dot(test_x[i], A[i])

    new_rep = new_rep / view_num

    X_W = np.dot(new_rep, W)

    # get the max
    predict = np.argmax(X_W, axis=1)

    return predict


def get_tree_group():
    file = 'tree_group.pkl'
    f = open(file, 'rb')

    tree_group = pickle.load(f, encoding='latin1')

    for i in tree_group:
        len = i.__len__()
        for j in range(len):
            i[j] -= 1

    f.close()

    return tree_group

def get_group_weight():
    file = 'tree_weight.pkl'
    f = open(file, 'rb')

    tree_weight = pickle.load(f, encoding='latin1')

    f.close()

    return tree_weight


def test_matrix():
    samples = 300000
    dim = 5000
    A = np.random.rand(samples, dim)

    start_time = timeit.default_timer()
    A_s = np.dot(A.T, A)
    print(A_s.shape)
    end_time = timeit.default_timer()

    print('Running time is %f' % (end_time - start_time))


def ones(ev):
    new_ev = []
    for item in range(188):
        item = 1.0
        new_ev.append(item)
    return new_ev


def main():
    Gv = get_tree_group()
    ev = get_group_weight()
    #     ev = ones(ev)

    file = 'leaf_mf_dataset.h5'
    f = tables.openFile(file, 'r')
    # the indices of multi-view feature
    view_indices = [0, 4096, 4296, 4396]

    root = f.root
    features = root.features
    labels = root.labels

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    # Provides train/test indices to split data in train test sets.
    sss = StratifiedShuffleSplit(labels, n_iter=5, test_size=0.2, random_state=0)

    macro_p_list = []
    macro_r_list = []
    macro_f_list = []
    micro_f_list = []

    # for each fold
    for train_index, test_index in sss:
        y_train, y_test = labels[train_index], labels[test_index]
        X_train, X_test = features[train_index], features[test_index]

        train_set_x = split_view(X_train, view_indices)
        test_set_x = split_view(X_test, view_indices)

        train_num = X_train.shape[0]

        task_indices = np.unique(y_train)
        task_dim = task_indices.__len__()
        if task_dim != np.unique(y_train).__len__():
            print('task dim error')

        Y = np.zeros((train_num, task_dim), dtype=np.int32)
        for i in range(task_dim):
            indices = (y_train == task_indices[i])
            Y[indices, i] = 1

        # parameters
        iteration_num = 20
        k = 200
        l1 = 0.1
        l2 = 0.1
        l3 = 0.2

        A, B, W = our_optimize(train_set_x, Y, k, Gv, ev, iteration_num, l1, l2, l3)

        # visualization of W
        #         plot_w(W)

        y_hat = mtl_class_predict(test_set_x, W, A)
        y_hat = np.array(y_hat, dtype=np.int32)

        f1_micro = f1_score(y_test, y_hat, average='micro')
        f1_macro = f1_score(y_test, y_hat, average='macro')
        accuracy = accuracy_score(y_test, y_hat)
        cf_matrix = confusion_matrix(y_test, y_hat)
        macro_p = precision_score(y_test, y_hat, average='macro')
        macro_r = recall_score(y_test, y_hat, average='macro')

        macro_p_list.append(macro_p)
        macro_r_list.append(macro_r)
        macro_f_list.append(f1_macro)
        micro_f_list.append(f1_micro)

        #         print('confusion_matrix is : ', cf_matrix)
        print('the micro f1 score of our model is : %f' % f1_micro)
        print('the macro f1 score of our model is : %f, %f, %f' % (macro_p, macro_r, f1_macro))
        print('the accuracy score of our model is : %f' % accuracy)

    print('the average macro precision score of our model is : %f' % np.mean(macro_p_list))
    print('the average macro recall score of our model is : %f' % np.mean(macro_r_list))
    print('the average macro f1 score of our model is : %f' % np.mean(macro_f_list))
    print('the average micro f1 score of our model is : %f' % np.mean(micro_f_list))

    print('the standard deviation macro precision score of our model is : %f' % np.std(macro_p_list))
    print('the standard deviation macro recall score of our model is : %f' % np.std(macro_r_list))
    print('the standard deviation macro f1 score of our model is : %f' % np.std(macro_f_list))
    print('the standard deviation micro f1 score of our model is : %f' % np.std(micro_f_list))


def our_model():
    Gv = get_tree_group()
    ev = get_group_weight()
    #     ev = ones(ev)

    file = 'leaf_mf_dataset.h5'

    dataset = load_data_file(file, False)

    view_indices = [0, 4096, 4296, 4396]
    #     view_indices = [4096,4296]
    #     view_indices = [0,4096]

    train_set, valid_set, test_set = dataset

    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set

    #     train_set_y += 1
    #     valid_set_y += 1
    #     test_set_y += 1

    train_num = train_set_x.shape[0]

    train_set_x = split_view(train_set_x, view_indices)
    valid_set_x = split_view(valid_set_x, view_indices)
    test_set_x = split_view(test_set_x, view_indices)

    # tasks
    task_indices = np.unique(train_set_y)
    task_dim = task_indices.__len__()
    if task_dim != np.unique(valid_set_y).__len__():
        print('task dim error')

    Y = np.zeros((train_num, task_dim), dtype=np.int32)
    for i in range(task_dim):
        indices = (train_set_y == task_indices[i])
        Y[indices, i] = 1

    # parameters
    iteration_num = 10

    k = 250

    l1 = 0.1
    l2 = 0.1
    l3 = 0.2

    A, B, W = our_optimize(train_set_x, Y, k, Gv, ev, iteration_num, l1, l2, l3)

    y_hat = mtl_class_predict(test_set_x, W, A)
    y_hat = np.array(y_hat, dtype=np.int32)

    f1_micro = f1_score(test_set_y, y_hat, average='micro')
    f1_macro = f1_score(test_set_y, y_hat, average='macro')
    accuracy = accuracy_score(test_set_y, y_hat)
    cf_matrix = confusion_matrix(test_set_y, y_hat)
    macro_p = precision_score(test_set_y, y_hat, average='macro')
    macro_r = recall_score(test_set_y, y_hat, average='macro')

    #         print('confusion_matrix is : ', cf_matrix)
    print('the micro f1 score of our model is : %f' % f1_micro)
    print('the macro f1 score of our model is : %f, %f, %f' % (macro_p, macro_r, f1_macro))
    print('the accuracy score of our model is : %f' % accuracy)


if __name__ == '__main__':
    our_model()
    #     main()

