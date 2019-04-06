
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


# In[392]:


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


# In[393]:


#Initializing values and computing H. Note the 1. to force to float type
m,n = x_train.shape
y_new = np.outer(y_train,y_train)

d = squareform(pdist(x_train, 'sqeuclidean'))
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


# In[394]:


#Setting solver parameters (change default to decrease tolerance) 
cvxopt_solvers.options['show_progress'] = False
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10


# In[395]:


#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.ravel(sol['x'])


# In[396]:


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


# In[397]:


true_out = np.array((pd.read_csv('orig_DHC_target_labels.txt', header = None, na_filter = False, low_memory = False)).values)


# In[398]:


ans = (np.array(predict)).reshape((600,1))


# In[399]:


print(1-(np.sum(abs(ans-true_out))/600))


# In[ ]:


x_output = sys.argv[3]
actual_path_out = os.path.abspath(x_output)
path_out = os.path.dirname(actual_path_out)
os.chdir(path_out)

np.savetxt(x_output, ans)

