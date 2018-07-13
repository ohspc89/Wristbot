# Python translation of Meirovitch's kinematic filter
# Version 1.0 Jinseok Oh

import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.integrate import solve_bvp

# bc() defines the boundary conditions of maximally smooth movement
# generation (jerk minimizing). It can be used with bcvc4c() and
# jerk_acc()
#
# BC: evaluates the residue of the boundary condition

def bc(varargin):

    if len(varargin) == 6:
        y0, y1, r0, r1, v0, v1 = varargin[:]
        res = np.array([y0[0] - r0, y0[1] - v0, y1[0] - r1, y1[1] - v1])
    elif len(varargin) == 8:
        y0, y1, r0, r1, v0, v1, a0, a1 = varargin[:]
        res = np.array([y0[0] - r0, y0[1] - v0, y0[2] - a0, y1[0] - r1, y1[1] - v1, y1[2] - a1])
    elif len(varargin) == 10:
        y0, y1, r0, r1, v0, v1, a0, a1 = varargin[:]
        res = np.array([y0[0] - r0, y0[1] - v0, y0[2] - a0, y0[3] - j0, y1[0] - r1, y1[1] - v1, y1[2] - a1, y1[3] - j1])

    return res

# MSDAccuracy_DE represents the differential equation for maximally smooth
# movements (jerk minimizing) and can be used with Matlab's bvc4c() to
# solve for the optimal generation of a trajectory

def MSDAccuracy_DE(t, y, lambda1, funct, direction):

    p = len(y)
    target = funct(t)
    dydt = np.vstack([y[1:], direction * lambda1 ** p * (y[0] - target)])

    return dydt

''' filter_JA(...) applies optimal Jerk (third derivative of position) filter to the input trajectory trj.
    'trj' is nxd array representing trajectory to be smoothed.
    'tt' is nx1 array of time points corresponding to 'trj'. If tt=[] then filter_JA assumes uniform sampling and uses tt=linspace(data)
    'l' is the accuracy demand (lambda)
    'endpoints' (optional) is a 2xd vector with the two endpoints [xstart,ystart, ...; xend,yend, ...]
    'vel' (optional) is a 2xd vector with the endpoint velocities
    [vxstart,xystart; vxend,vyend]
    'acc' (optional) is  a vector with the endpoint accelerations [axstart,aystart,...; axend,ayend, ...]
    'direction' ('optional') determines if the system is attracted to 'trj'
    (direction=1, default) or repelled by 'trj'.
    method (optional) is a string: 'slow' for a seond recomputation of optimal trajectory using a first solution as initial guess.
    'fast' (default) for one solution based on initial guess only.
    varargout = [x, y,...  solx, soly,...]

    Copyright (C) Yaron Meirovitch, 2012-2014 '''

def filter_JA(trj, lambda1 = None, tt = np.array([]), endpoints = np.array([[],]), vel = None, acc = None, direction = None, method = "fast"):

    if not lambda1:
        lambda1 = np.ceil(trj.size/20)

    d = np.size(trj, axis = 1)
    l = lambda1 * np.size(trj, axis = 0) / 250 * (d**2 / 4)

    # a sample structure of endpoints would be something like the one given as default.
    if np.size(endpoints, axis = 0) == 1:
        endpoints = endpoints.reshape(d,-1)

    # don't know of any good data type that replaces <cell> in MATLAB could be replaced later
    varargout = dict()

    if not direction:
        direction = l

    if tt.size == 0:
        tt = np.linspace(0, 1, np.size(trj, axis = 0))

    for i in range(d):
        ''' for MATLAB's function for 1D interpolation, <griddedInterpolant>,
         there are several python alternatives; two of the applicable alternatives
         are <interp1d> and <InterpolatedUnivariateSpline>. I have wrote down the
         codes so that the user could apply one of the two.'''

        # k = 3 is cubic spline
        F1 = InterpolatedUnivariateSpline(tt[~np.isnan(trj[:,i])], trj[~np.isnan(trj[:,i]), i], k = 3)

        # or it could be done in this fashion
        # F1 = interp1d(tt[~np.isnan(trj[:,i])], trj[~np.isnan(trj[:,i]), i], fill_value = "extrapolate")

        if not endpoints.any():
            rx0 = F1(tt[0]); rx1 = F1(tt[-1])
        else:
            rx0 = endpoints[0, i]; rx1 = endpoints[1, i]

        if not vel:
            vx0, vx1 = [0, 0]
        else:
            vx0 = vel[0, i]; vx1 = vel[1,i]

        if not acc:
            ax0, ax1 = [0, 0]
        else:
            ax0 = acc[0, i]; ax1 = acc[1,i]

        bcx = lambda y0, y1: bc([y0, y1, rx0, rx1, vx0, vx1, ax0, ax1])

        bvpx = lambda t, y: MSDAccuracy_DE(t, y, l, F1, direction)

        t1 = tt[0]; t2 = tt[-1]
        x0 = rx0 - F1(t1); xf = rx1 - F1(t2)
        a0 = x0 + (t2 * (5 * t1 ** 4 * x0 - 5 * t1 ** 4 * xf) - t1 ** 5 * x0 + t1 ** 5 * xf - t2 ** 2 * (10 * t1 ** 3 * x0 - 10 * t1 ** 3 * xf)) / (t1 - t2) ** 5
        a1 = (30 * t1 ** 2 * t2 ** 2 * (x0 - xf)) / (t1 - t2) ** 5
        a2 = -(30 * t1 * t2 * (t1 + t2) * (x0 - xf)) / (t1 - t2) ** 5
        a3 = (10 * (x0 - xf) * (t1 ** 2 + 4 * t1 * t2 + t2 ** 2)) / (t1 - t2) ** 5
        a4 = -(15 * (t1 + t2) * (x0 - xf)) / (t1 - t2) ** 5
        a5 = (6 * (x0 - xf)) / (t1 - t2) ** 5

        guess_funx = lambda t : [F1(t) + a5 * t ** 5 + a4 * t ** 4 + a3 * t ** 3 + a2 * t ** 2 + a1 * t + a0,\
                5 * a5 * t ** 4 + 4 * a4 * t ** 3 + 3 * a3 * t ** 2 + 2 * a2 * t + a1,\
                20 * a5 * t ** 3 + 12 * a4 * t ** 2 + 6 * a3 * t + 2 * a2,\
                60 * a5 * t ** 2 + 24 * a4 * t + 6 * a3,\
                24 * a4 + 120 * a5 * t,\
                120 * a5]

        guessx2 = np.array(list(map(guess_funx, tt)))
        res = solve_bvp(bvpx, bcx, tt, guessx2.transpose(), tol = 2)

        if method == "slow":
            res = solve_bvp(bvpx, bcx, tt, res.sol(tt), tol = 2)

        varargout[i] = res.sol(tt).transpose()
        varargout[i+d] = res

    return varargout

