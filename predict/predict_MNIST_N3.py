# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 03:31:06 2018

@author: zfang
"""

import numpy as np
import scipy.io
import os.path

num_init = 50
DH = 20

M = range(1000, 10100)
Yin = np.load(r'C:\Users\zfang\Downloads\data\imtrain_noisy_[1, 7].npy')[M]
Yout = np.load(r'C:\Users\zfang\Downloads\data\labtrain_noisy_[1, 7].npy')[M]

Xh = np.zeros((len(M), DH), dtype = np.float64)
Xout = np.zeros((len(M), 2), dtype = np.float64)

accuracy = np.transpose(np.matrix([[], []], dtype=np.float64))

def sigmoid(X, W, b):
    linpart = np.matmul(X, W) + b
    return 1.0 / (1.0 + np.exp(-linpart))

for i in xrange(0, num_init):
    
    WPATH = r'E:\GoogleDrive\0. DAML\VarAnneal and MNIST\0. MNIST-Lite [1][7]\N3\DH' + str(DH) + r'_100ex\W_' + str(i + 1) + '.npy'
    bPATH = r'E:\GoogleDrive\0. DAML\VarAnneal and MNIST\0. MNIST-Lite [1][7]\N3\DH' + str(DH) + r'_100ex\b_' + str(i + 1) + '.npy'
    
    if os.path.isfile(WPATH):
        W = np.load(WPATH)
        b = np.load(bPATH)
    
        for j in xrange(2):
            for k in xrange(DH):
                Xh[:, k] = sigmoid(Yin, W[-1, 0][k], b[-1, 0][k])
        
            Xout[:, j] = sigmoid(Xh, W[-1, 1][j], b[-1, 1][j])
        
        pred = np.rint(Xout[:, 0])
        label = np.rint(Yout[:, 0])
        
        acc = np.matrix([i + 1, np.mean(pred == label)])
        accuracy = np.concatenate((accuracy, acc), axis = 0)

scipy.io.savemat(r'accuracy_DH' + str(DH) + r'_100ex.mat', dict(accuracy = accuracy))
