
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import math
import os
import seaborn as sns
import time

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

class threevariables(twovariables):

    def __init__(self, filename):
        twovariables.__init__(self, filename)
        self.S = int(self.params[2][1:])

class PIDobject(threevariables, twovariables):

    def __init__(self,filename):
        if len(filename.split(',')) == 3:
            threevariables.__init__(self, filename)
        else:
            twovariables.__init__(self, filename)

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
        plt.figure(1)
        plt.plot(self.get_torques(axis))
        plt.title("P{}, D{}, S{}, {}".format(self.P, self.D, self.S, torquename), fontweight = "bold")
        plt.xlabel("Index")
        plt.ylabel(torquename)
        plt.show()

'''this is the fuction that will create and store all the PIDobjects'''
def create_PIDs(directory):
    pidobjs = list()

    if os.getcwd() != directory:
        os.chdir(directory)

    for filename in os.listdir(directory):
        if filename.endswith("txt"):
            pidobjs.append(PIDobject(filename))

    return pidobjs

'''For error calculation'''
def PIDobjs_mean_avg_err(pidobjs):
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

    return err_df

'''plotting a heatmap for the mean errors'''
'''The function allows to set which parameter value to sit at which axis of the heatmap plot'''
def mean_avg_err_heatmap(df, xaxis = "P", yaxis = "D"):
    df4heatmap = df[df.S == 100].pivot(yaxis, xaxis, "error")

    plt.figure(1)
    sns.heatmap(df4heatmap)
    plt.title('A heatmap of mean absolute errors for PID values', fontweight = 'bold')
    plt.show()

'''The function which_torqueplot will receive pidobjs, P, D, S values and the axis at which the torque_values will be plotted'''
'''The default torque axis is x-axis'''
def which_torqueplot(pidobjs, P, D, S, axis="x"):
    for pidobj in pidobjs:
        if (pidobj.P == P) & (pidobj.D == D) & (pidobj.S == S):
            pidobj.plot_torques(axis)

if __name__ == "__main__":
    directory = input("Type your directory: ")
    print('You typed:', directory)

    pidobjs = create_PIDs(directory)

    cont_yes = ''
    while cont_yes != 'n':
        activity = ''
        while activity not in ['0', '1']:
            activity = input("Choose your activity:\n0: plot a heatmap of the mean absolute errors\n1: plot torque values\n")
            if activity == '0':
                mean_errs = PIDobjs_mean_avg_err(pidobjs)
                mean_avg_err_heatmap(mean_errs)
            else:
                P = int(input("P: "))
                D = int(input("D: "))
                S = int(input("S: "))
                axis = input("axis: ")
                which_torqueplot(pidobjs, P, D, S, axis)
        cont_yes = input("Do you want to continue?[y/n]: ")
