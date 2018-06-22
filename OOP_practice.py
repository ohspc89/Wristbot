
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
        self.P = float(self.params[0][1:])
        self.D = float(self.params[1][1:])
        self.S = float(100)
        self.df = self.read_file()

    '''read the file with the filename'''
    def read_file(self):
        file = pd.read_csv(self.filename, sep = '\t')
        return file

class threevariables(twovariables):

    def __init__(self, filename):
        twovariables.__init__(self, filename)
        self.S = float(self.params[2][1:])

class PIDobject(threevariables, twovariables):

    def __init__(self,filename):
        if len(filename.split(',')) == 3:
            threevariables.__init__(self, filename)
        else:
            twovariables.__init__(self, filename)

    def plot_diff(self):
        plt.figure()
        plt.plot(self.df.target_x)
        plt.plot(self.df.positn_x, alpha = 0.8)
        plt.title("P{},D{},S{}, difference in positions".format(self.P, self.D, self.S), fontweight = "bold")
        plt.xlabel("Index")
        plt.ylabel("Distance in degrees")

    def get_turn_pofloats(self):
        return np.where(np.diff(self.df.target_x) != 0)[0]

    def make_windows(self):
        turn_pofloats = self.get_turn_pofloats()
        start = turn_pofloats[ : -1] + 1
        end = turn_pofloats[1 : ] + 1
        return [self.df.iloc[a : b] for a, b in zip(start, end)]

    '''add an option to choose which type of error to look at'''
    '''the types are: total, targets and origins'''
    def abs_err(self, option = "total"):
        windows = self.make_windows()

        positn_xs = [np.unique(window.positn_x, return_counts = True) for window in windows]
        error = np.array([abs(max(window.target_x) - positn_xs[i][0][positn_xs[i][1].argmax()]) for i, window in enumerate(windows)])

        targets = np.array([error[i] for i in range(len(error)) if i % 2 == 0])
        origins = error[~np.isin(error, targets)]

        if option == "targets":
            return np.array([x * 180 / math.pi for x in targets])
        elif option == "origins":
            return np.array([x * 180 / math.pi for x in origins])
        else:
            return np.array([x * 180 / math.pi for x in error])

    def get_torques(self, axis = 'x'):
        torquename = "torque_" + axis
        return self.df[torquename]

    def plot_torques(self, axis = "x"):
        torquename = "torque_" + axis
        plt.figure()
        plt.plot(self.get_torques(axis))
        plt.title("P{}, D{}, S{}, {}".format(self.P, self.D, self.S, torquename), fontweight = "bold")
        plt.xlabel("Index")
        plt.ylabel(torquename)
        plt.show(block=False)

'''this is the fuction that will create and store all the PIDobjects'''
def create_PIDs(directory):
    pidobjs = list()

    if os.getcwd() != directory:
        os.chdir(directory)

    for filename in os.listdir(directory):
        if filename.endswith("txt"):
            pidobjs.append(PIDobject(filename))

    return pidobjs

'''plot difference between positn_x and target_x'''
def which_diffplot(pidobjs, P, D, S = 100):
    for pidobj in pidobjs:
        if (pidobj.P == P) & (pidobj.D == D) & (pidobj.S == S):
            pidobj.plot_diff()

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

    plt.figure()
    sns.heatmap(df4heatmap)
    plt.title('A heatmap of mean absolute errors for PID values', fontweight = 'bold')
    plt.show(block = False)

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
        while activity not in ['0', '1', '2']:
            activity = input("Choose your plot type:\n0: distance between the target and the \
actual position\n1: a heatmap of the mean absolute errors\n2: torque values(x, y, z)\n")
            if activity == '0':
                P = input("P: ")
                D = input("D: ")
                S = input("S (hit enter if it is 100): ")
                if S == '':
                    S = 100
                which_diffplot(pidobjs, float(P), float(D), float(S))

            elif activity == '1':
                mean_errs = PIDobjs_mean_avg_err(pidobjs)
                mean_avg_err_heatmap(mean_errs)

            else:
                P = input("P: ")
                D = input("D: ")
                S = input("S (hit enter if it is 100): ")
                if S == '':
                    S = 100
                axis = input("axis: ")
                which_torqueplot(pidobjs, float(P), float(D), float(S), axis)
        cont_yes = input("Do you want to continue?[y/n]: ")
    print("Bye bye!")
