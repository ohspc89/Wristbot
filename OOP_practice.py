
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import math
import os
import seaborn as sns

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
        plt.show()

    def get_turn_points(self):
        return np.where(np.diff(self.df.target_x) != 0)[0]

    def make_windows(self):
        turn_points = self.get_turn_points()
        start = turn_points[ : -1] + 1
        end = turn_points[1 : ] + 1
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
            return np.array([x * 180 / np.pi for x in targets])
        elif option == "origins":
            return np.array([x * 180 / np.pi for x in origins])
        else:
            return np.array([x * 180 / np.pi for x in error])

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
    plt.show()

'''The function which_torqueplot will receive pidobjs, P, D, S values and the axis at which the torque_values will be plotted'''
'''The default torque axis is x-axis'''
def which_torqueplot(pidobjs, P, D, S, axis="x"):
    for pidobj in pidobjs:
        if (pidobj.P == P) & (pidobj.D == D) & (pidobj.S == S):
            pidobj.plot_torques(axis)

def print_error_all_or_not(pidobjs, all_yes):
    '''ask if the data is from Joint Matching Task'''
    isjmt = input("Is your data from Joint Matching Task?:\n0: No\n1: Yes\n")

    '''loop until we get a correct response for [isjmt]'''
    while isjmt not in ['0', '1']:
        isjmt = input("Wrong number, choose again.\nIs your data from Joint Matching Task?:\n0: No\n1: Yes\n")

    if isjmt == "1":
        err_type = input("Choose the type of distance :\n0: All distances\n1: Distnaces at origins\n2: Distances at targets(passive movement)\n3: Distances at targets(active movement)\n")
        while err_type not in ['0', '1', '2', '3']:
            err_type = input("Wrong number, choose again.\nChoose the type of distance :\n0: All distances\n1: Distances at origins\n2: Distances at targets(passive movement)\n3: Distances at targets(active movement)\n")
    else:
        err_type = input("Choose the type of distance :\n0: All distances\n1: Distnaces at origins\n2: Distances at targets\n")
        while err_type not in ['0', '1', '2']:
            err_type = input("Wrong number, choose again.\nChoose the type of distance :\n0: All distances\n1: Distances at origins\n2: Distances at targets\n")

    if all_yes == "y":
        for pidobj in pidobjs:
            print_error(pidobj, isjmt, err_type)
    else:
        P = input("P: ")
        D = input("D: ")
        S = input("S (hit enter if it is 100): ")
        if S == '':
            S = 100
        for pidobj in pidobjs:
            if (pidobj.P == float(P)) & (pidobj.D == float(D)) & (pidobj.S == float(S)):
                print_error(pidobj, isjmt, err_type)

'''added a parameter [isjmt] to indicate if the data is from joint matching task'''
def print_error(pidobj, isjmt, err_type):
    if err_type == "0":
        print("P {}, D {}, S {}".format(pidobj.P, pidobj.D, pidobj.S) + ": mean = %f var = %f" % (np.average(pidobj.abs_err()), np.var(pidobj.abs_err())))
    elif err_type == '1':
        print("P {}, D {}, S {}".format(pidobj.P, pidobj.D, pidobj.S) + ": mean = %f var = %f" % (np.average(pidobj.abs_err('origins')), np.var(pidobj.abs_err('origins'))))
    elif err_type == "2":
        '''if the data is from JMT, err_type # 2 is the error at target positions, passive movements'''
        if isjmt == "1":
            target_pm = np.array([x for i, x in enumerate(pidobj.abs_err("targets")) if i % 2 == 0])
            print("P {}, D {}, S {}".format(pidobj.P, pidobj.D, pidobj.S) + ": mean = %f var = %f" % (np.average(target_pm), np.var(target_pm)))
        '''if the data is NOT from JMT, err_type # 2 is error at target position and no indication of passive or active'''
        else:
            print("P {}, D {}, S {}".format(pidobj.P, pidobj.D, pidobj.S) + ": mean = %f var = %f" % (np.average(pidobj.abs_err('targets')), np.var(pidobj.abs_err('targets'))))
    '''now, err_type # 3 is the error at target positions, active movements'''
    else:
        target_am = np.array([x for i, x in enumerate(pidobj.abs_err("targets")) if i % 2 == 1])
        print("P {}, D {}, S {}".format(pidobj.P, pidobj.D, pidobj.S) + ": mean = %f var = %f" % (np.average(target_am), np.var(target_am)))

if __name__ == "__main__":
    directory = input("Type your directory: ")
    print('You typed:', directory)

    pidobjs = create_PIDs(directory)

    cont_yes = ''
    while cont_yes != 'n':
        activity = ''
        while activity not in ['0', '1', '2', '3']:
            activity = input("What do you want to do? :\n0: get the means and the variances of the distances between the targets and \
the actual positions\n1: plot the distances\n2: plot a heatmap of the mean absolute errors\n3: torque values(x, y, z)\n")
            if activity == '0':
                all_yes = input("Do you want to print the means and the variances of all the data?[y/n]: ")
                while all_yes not in ['y', 'n']:
                    all_yes = input("Wrong Input!!\nDo you want to print the means and the variances of all the data?[y/n]: ")
                print_error_all_or_not(pidobjs, all_yes)

            elif activity == '1':
                P = input("P: ")
                D = input("D: ")
                S = input("S (hit enter if it is 100): ")
                if S == '':
                    S = 100
                which_diffplot(pidobjs, float(P), float(D), float(S))

            elif activity == '2':
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
