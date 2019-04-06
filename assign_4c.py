
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy as sp
from scipy.spatial.distance import pdist, squareform

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

import os
import sys


# In[2]:


x_train_path = sys.argv[1]
train_path = os.path.abspath(x_train_path)
path_train = os.path.dirname(train_path)
os.chdir(path_train)

x_test_path = sys.argv[2]
test_path = os.path.abspath(x_test_path)
path_test = os.path.dirname(test_path)
os.chdir(path_test)

x_train = (pd.read_csv(x_train_path, header = None, na_filter = False, low_memory = False)).values
x_test = (pd.read_csv(x_test_path, header = None, na_filter = False, low_memory = False)).values

y_train = np.array(x_train[:,0])

x_train = np.array(x_train[:,1:], dtype = np.float32)
x_test = np.array(x_test[:,1:], dtype = np.float32)

x_train = x_train/255.
x_test = x_test/255.

y_train[y_train == 0] = -1
y_train.reshape((y_train.shape[0],1))

C = float(sys.argv[4])
gamma = float(sys.argv[5])
threshold = 1e-4


# In[3]:


x_train_zero = x_train - np.mean(x_train, axis = 0)
M = np.cov(x_train_zero.T)

lamda, v = np.linalg.eig(M)

idx = lamda.argsort()[::-1]

lamda = lamda[idx]
v = v[:,idx]

lamda = lamda[:50]
v = v[:,:50]

# print(lamda.shape)
# print(v.shape)

# print(x_train.shape)
# print(x_test.shape)

x_train_new = (v.T).dot(x_train.T).T
x_test_new = (v.T).dot(x_test.T).T

# print(x_train_new.shape)
# print(x_test_new.shape)


# In[2]:


# from sklearn.decomposition import RandomizedPCA
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# pca = PCA(100)
# pca.fit(x_train[0:100].data)

# fig, axes = plt.subplots(2, 4, figsize=(8, 8),
#                          subplot_kw={'xticks':[], 'yticks':[]},
#                          gridspec_kw=dict(hspace=0.1, wspace=0.1))

# for i, ax in enumerate(axes.flat):
#     ax.imshow(pca.components_[i].reshape(32, 32), cmap='gray')


# In[1]:


# from pylab import *

# fig, axes = plt.subplots(2, 4, figsize=(8, 8),
#                          subplot_kw={'xticks':[], 'yticks':[]},
#                          gridspec_kw=dict(hspace=0.1, wspace=0.1))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(x_train[i,:].reshape((32,32)), cmap='gray')


# In[6]:


#Initializing values and computing H. Note the 1. to force to float type
m,n = x_train_new.shape
y_new = np.outer(y_train,y_train)

d = squareform(pdist(x_train_new, 'sqeuclidean'))
k = sp.exp(-gamma*d)

H = y_new * k

#Converting into cvxopt format
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(np.vstack((np.diag(-np.ones(m)), np.identity(m))))
h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m)*C)))
A = cvxopt_matrix(y_train.reshape(1, -1))
A = cvxopt_matrix(A, (1, m), 'd')
b = cvxopt_matrix(np.zeros(1))


# In[7]:


#Setting solver parameters (change default to decrease tolerance) 
cvxopt_solvers.options['show_progress'] = False
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10


# In[8]:


#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.ravel(sol['x'])


# In[9]:


S = np.where(alphas > threshold)[0][0]
temp = alphas*y_train

asd = sp.exp(-gamma*np.linalg.norm(x_train-x_train[S],axis=1))
b = y_train[S] - asd @ temp.T
predict = []

for i in range(x_test.shape[0]):
    asv = sp.exp(-gamma*np.linalg.norm(x_train-x_test[i],axis=1))
    wx = asv @ temp.T
    pred = np.sign(wx+b)
    if(pred < 0):
        predict.append(0)
    else:
        predict.append(1)


# In[10]:


true_out = np.array((pd.read_csv('orig_DHC_target_labels.txt', header = None, na_filter = False, low_memory = False)).values)


# In[11]:


ans = (np.array(predict)).reshape((600,1))


# In[12]:


print(1-(np.sum(abs(ans-true_out))/600))


# In[ ]:


x_output = sys.argv[3]
actual_path_out = os.path.abspath(x_output)
path_out = os.path.dirname(actual_path_out)
os.chdir(path_out)

np.savetxt(x_output, ans)

