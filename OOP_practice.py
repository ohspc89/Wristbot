
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import math
import os
from datetime import datetime
import seaborn as sns

class twovariables:

    def __init__(self, filename):
        self.filename = filename
        self.params = filename[:-4].split(',')
        self.P = float(self.params[0][1:])
        self.D = float(self.params[1][1:])
        self.S = float(100)
        self.df = self.read_file()

    # read the file with the filename
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
        plt.plot(self.df.target_x * 180 / np.pi)
        plt.plot(self.df.positn_x * 180 / np.pi, alpha = 0.8)
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

    # add an option to choose which type of error to look at
    # the types are: total, targets and origins
    def abs_err(self, option = "total"):
        windows = self.make_windows()

        positn_xs = [np.unique(window.positn_x, return_counts = True) for window in windows]
        error = np.array([abs(max(window.target_x) - positn_xs[i][0][positn_xs[i][1].argmax()]) for i, window in enumerate(windows)])

        targets = np.array([error[i] for i in range(len(error)) if i % 2 == 0])
        origins_full = error[~np.isin(error, targets)]
        origins = np.array([origins_full[i] for i in range(len(origins_full)) if i % 2 != 0])

        if option == "targets":
            return targets * 180 / np.pi
        elif option == "origins":
            return origins * 180 / np.pi
        else:
            return error * 180 / np.pi

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

# this is the fuction that will create and store all the PIDobjects
def create_PIDs(directory):
    pidobjs = list()

    if os.getcwd() != directory:
        os.chdir(directory)

    for filename in os.listdir(directory):
        if filename.endswith("txt"):
            pidobjs.append(PIDobject(filename))

    return pidobjs

# plot difference between positn_x and target_x
def which_diffplot(pidobjs, P, D, S = 100):
    for pidobj in pidobjs:
        if (pidobj.P == P) & (pidobj.D == D) & (pidobj.S == S):
            pidobj.plot_diff()

# For error calculation
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

# The function which_torqueplot will receive pidobjs, P, D, S values and the axis at which the torque_values will be plotted; default = x
def which_torqueplot(pidobjs, P, D, S, axis="x"):
    for pidobj in pidobjs:
        if (pidobj.P == P) & (pidobj.D == D) & (pidobj.S == S):
            pidobj.plot_torques(axis)

# A function to get the user inputs for P, I, D values
def get_PID_vals():
    print("Remember, if your type parameter values are absent, you will get no result")
    P = input("P: ")
    D = input("D: ")
    S = input("S (hit enter if it is 100): ")
    if S == '':
        S = 100
    return P, D, S


def print_error_all_or_not(pidobjs, all_yes):
    # ask if the data is from Joint Matching Task
    isjmt = yes_or_no("jmt")

    if isjmt in ["y", "Y"]:
        while True:
            try:
                err_type_idx = input("Choose the type of distance :\n0: All distances\n1: Distnaces at origins\n2: Distances at targets(passive movement)\n3: Distances at targets(active movement)\n")
                if err_type_idx not in ["0", "1", "2", "3"]:
                    raise ValueError("Wrong input, type again")
                else:
                    break
            except ValueError as e:
                print(e)

    else:
        while True:
            try:
                err_type_idx = input("Choose the type of distance :\n0: All distances\n1: Distance at origins\n2: Distance at targets\n")
                if err_type_idx not in ["0", "1", "2"]:
                    raise ValueError("Wrong input, type again")
                else:
                    break
            except ValueError as f:
                print(f)

    # Define the output file name
    filename = datetime.now().strftime('PID_log_%H:%M:%S_%m-%d.csv')
 
    if all_yes in ["y", "Y"]:
        for pidobj in pidobjs:
            print_error(pidobj, isjmt, err_type_idx, filename)
    else:
        P, D, S = get_PID_vals()
        for pidobj in pidobjs:
            if (pidobj.P == float(P)) & (pidobj.D == float(D)) & (pidobj.S == float(S)):
                print_error(pidobj, isjmt, err_type_idx, filename)

# added a parameter [isjmt] to indicate if the data is from joint matching task
def print_error(pidobj, isjmt, err_type_idx, filename):

    P = pidobj.P; D = pidobj.D; S = pidobj.S

    if err_type_idx == "0":
        print_in_format(P, D, S, pidobj.abs_err(), filename)

    elif err_type_idx == "1":
        print_in_format(P, D, S, pidobj.abs_err("origins"), filename)

    elif err_type_idx == "2":
        if isjmt in ["y", "Y"]:
            target_pm = np.array([x for i, x in enumerate(pidobj.abs_err("targets")) if i % 2 == 0])
            print_in_format(P, D, S, target_pm, filename)
        else:
            print_in_format(P, D, S, pidobj.abs_err("targets"), filename)

    else:
        target_am = np.array([x for i, x in enumerate(pidobj.abs_err("targets")) if i % 2 == 1])
        print_in_format(P, D, S, target_am, filename)

# Print the calculated error values to a csv file
def print_in_format(P, D, S, input_for_stat, filename):
    with open(filename, "a+") as csv:
        # The order of the printed figures should be P / D / S / error mean / error variance
        csv.write("{},{},{},{},{}\n".format(P, D, S, np.average(input_for_stat), np.var(input_for_stat)))

# a function to get y/n until the user types a correct one
def yes_or_no(type = "whole_again"):
    if type == "same_again":
        comment = "Do you want to do the same for some other file?[y/n]: "
    elif type == "jmt":
        comment = "Is your data coming from Joint Matching Task?[y/n]: "
    elif type == "all_file":
        comment = "Do you want to print the means and the variances of all the data?[y/n]: "
    else:
        comment = "Do you want to continue?[y/n]: "
    while True:
        try:
            yesno = input(comment)
            if yesno not in ["y", "Y", "n", "N"]:
                raise ValueError("Wrong Input, type again")
            else:
                break
        except ValueError as e:
            print(e)
    return yesno

if __name__ == "__main__":
    directory = input("Type your directory: ")
    print('You typed:', directory)

    pidobjs = create_PIDs(directory)

    while True:
        try:
            activity = input("What do you want to do?\n0: get the means and the variances of the distances between the targets and \
the actual positions\n1: plot the distances\n2: plot a heatmap of the mean absolute errors\n3: torque values(x, y, z)\n")
            if activity not in ['0', '1', '2', '3']:
                raise ValueError('Wrong Input, type again')

            elif activity == '0':
                while True:
                    all_yes = yes_or_no("all_file")
                    print_error_all_or_not(pidobjs, all_yes)
                    cont_same = yes_or_no("same_again")
                    if cont_same in ["y", "Y"]:
                        continue
                    else:
                        break

            elif activity == '1':
                while True:
                    P, D, S = get_PID_vals()
                    which_diffplot(pidobjs, float(P), float(D), float(S))
                    cont_same = yes_or_no("same_again")
                    if cont_same in ["y", "Y"]:
                        continue
                    else:
                        break

            elif activity == "2":
                mean_errs = PIDobjs_mean_avg_err(pidobjs)
                mean_avg_err_heatmap(mean_errs)

            else:
                while True:
                    P, D, S = get_PID_vals()
                    axis = input("Axis[x or y]: ")
                    which_torqueplot(pidobjs, float(P), float(D), float(S), axis)
                    cont_same = yes_or_no("same_again")
                    if cont_same in ["y", "Y"]:
                        continue
                    else:
                        break

            '''check if the user want to continue any process'''
            cont_proc = yes_or_no("whole_again")
            if cont_proc in ["y", "Y"]:
                continue
            else:
                print("Bye bye!")
                break

        except ValueError as h:
            print(h)


