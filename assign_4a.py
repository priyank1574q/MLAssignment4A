
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

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

threshold = 1e-4


# In[3]:


#Initializing values and computing H. Note the 1. to force to float type
m,n = x_train.shape
y = y_train.reshape(-1,1) * 1.
x_dash = y * x_train
H = np.dot(x_dash , x_dash.T) * 1.

#Converting into cvxopt format
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(np.vstack((np.diag(-np.ones(m)), np.identity(m))))
h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m)*C)))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))


# In[4]:


#Setting solver parameters (change default to decrease tolerance) 
cvxopt_solvers.options['show_progress'] = False
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10


# In[5]:


#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.ravel(sol['x'])


# In[6]:


S = np.where(alphas > threshold)[0][0]
temp = alphas*y_train

b = y_train[S] - ((x_train[S] @ x_train.T) @ temp.T)

wx = x_test @ (temp @ x_train)

predict = np.sign(wx+b)
predict[predict<0] = 0


# In[8]:


true_out = np.array((pd.read_csv('orig_DHC_target_labels.txt', header = None, na_filter = False, low_memory = False)).values)


# In[9]:


ans = (np.array(predict)).reshape((600,1))


# In[10]:


print(1-(np.sum(abs(ans-true_out))/600))


# In[ ]:


x_output = sys.argv[3]
actual_path_out = os.path.abspath(x_output)
path_out = os.path.dirname(actual_path_out)
os.chdir(path_out)

np.savetxt(x_output, ans)

