
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import math
import os
import seaborn as sns
import time


# In[56]:


filename = 'P+13,D-9.txt'
test = twovariables(filename)


# In[58]:


class twovariables:
    
    def __init__(self, filename):
        self.filename = filename
        self.params = filename[:-4].split(',')
        self.P = int(self.params[0][1:])
        self.D = int(self.params[1][1:])
        self.S = int(100)
        self.df = self.read_file()
    
    '''read the file with the filename'''
    def read_file(self):
        file = pd.read_csv(self.filename, sep = '\t')
        return file


# In[59]:


class threevariables(twovariables):
    
    def __init__(self, filename):
        twovariables.__init__(self, filename)
        self.S = int(self.params[2][1:])


# In[79]:


class PIDobject(threevariables, twovariables):
    
    def __init__(self,filename):
        if len(filename.split(',')) == 3:
            threevariables.__init__(self, filename)
        else:
            twovariables.__init__(self, filename)

    '''this part needs to be fixed later... I want this function to be running without parameter <df>
    and still do not call self.read_file() twice'''
    
    '''I will just temporarily make two different functions: get_turn_points and make_turn_points'''
    '''NO THAT IS REDUNDANT, FIND ANOTHER WAY.'''
    def get_turn_points(self):
        return np.where(np.diff(self.df.target_x) != 0)[0]
    
    def make_windows(self):
        turn_points = self.get_turn_points()
        start = turn_points[ : -1] + 1
        end = turn_points[1 : ] + 1
        return [self.df.iloc[a : b] for a, b in zip(start, end)]
    
    def abs_err(self):   
        windows = self.make_windows()

        positn_xs = [np.unique(window.positn_x, return_counts = True) for window in windows]
        error = [abs(max(window.target_x) - positn_xs[i][0][positn_xs[i][1].argmax()]) for i, window in enumerate(windows)]

        return [x * 180 / math.pi for x in error]
    
    def get_torques(self, axis = 'x'):
        torquename = "torque_" + axis
        return self.df[torquename]
    
    def plot_torques(self, axis = "x", return_values = False):
        torquename = "torque_" + axis
        plt.figure()
        plt.plot(self.get_torques(axis))
        plt.title("P{}, D{}, S{}, {}".format(self.P, self.D, self.S, torquename), fontweight = "bold")
        plt.xlabel("Index")
        plt.ylabel(torquename)
        plt.show()
        if return_values == True:
            return self.torque_be_plotted


# In[73]:


axis = 'x'
axis.join('torque_')


# In[5]:


os.chdir(os.getcwd() + '/Wristbot PID Finetuning Data')


# In[6]:


directory = os.getcwd()


# In[80]:


start = time.time()
pidobjs = list()
for filename in os.listdir(directory):
    if filename.endswith("txt"):
        pidobjs.append(PIDobject(filename))

'''For error calculation'''

Ps = Ds = Ss = Es  = pd.Series()

for pidobj in pidobjs:
    Ps = Ps.append(pd.Series(pidobj.P))
    Ds = Ds.append(pd.Series(pidobj.D))
    Ss = Ss.append(pd.Series(pidobj.S))
    avgerr = np.average(pidobj.abs_err())
    Es = Es.append(pd.Series(avgerr))

err_df = pd.DataFrame(columns = ["P", "D", "S", "error"])
err_df['P'] = Ps
err_df['D'] = Ds
err_df['S'] = Ss
err_df['error'] = Es

err_df = err_df.sort_values(by = ["P", "D", "S"])
err_df.index = range(len(pidobjs))
time.time() - start


# In[24]:


df4heatmap = err_df[err_df.S == 100].pivot("D", "P", "error")
plt.figure(1)
sns.heatmap(df4heatmap)
plt.title('A heatmap of mean absolute errors for PID values', fontweight = 'bold')
plt.show()


# In[85]:


'''torque'''

params = [13, 4, 100]
pidobjs[(pidobj.P == params[0]) & (pidobj.D == params[1]) & (pidobj.S == params[2])].plot_torques("z")

