import pandas as pd
import numpy as np
from scipy import fftpack, signal
from scipy.signal import butter, lfilter, sosfilt, sosfreqz, sosfilt_zi, sosfiltfilt, convolve, hilbert, find_peaks
from scipy.signal._arraytools import odd_ext, axis_slice, axis_reverse
import matplotlib.pyplot as plt

class accel_data:
    
    def __init__(self, filename, sex = "M"):
        self.filename = filename
        self.df = self.read_file()
        self.name = [x for x in self.filename.strip().split("_") if not x.startswith('300') and not x.endswith('txt')][0]
        self.sex = sex
        
        # parameters
        self.fs = 1000.0 # sampling frequency
        self.T = 1.0 / self.fs # period
        self.N = self.df.shape[0] # number of samples
        self.nyq = self.fs / 2.0 # nyquist frequency: fs / 2
        self.lowcut, self.highcut = self.get_bandpass()
        
        # actual accel data
        self.accel_x = self.df.x
        self.accel_y = self.df.y
        self.accel_z = self.df.z
        
    def read_file(self):
        file = pd.read_csv(self.filename, skiprows = 5, header = None)
        file.rename(columns = {0:"x", 1:"y", 2:"z", 3:"trig"}, inplace = True)
        
        # remove where Nan is included
        file.drop(file.isnull().any(1).nonzero()[0], inplace = True)
        return file
    
    def fft_prep(self, accel_axis = "z"):       
        # T = 1/1000 = 0.001 means that the window between two data points
        # is worth 0.001 second.
        # Therefore, N * T means the total time of the signal
        x = np.linspace(0.0, self.N * self.T, self.N)
        y = getattr(self.df, accel_axis[:])

        # Fast Fourier Transform
        fft_y = fftpack.rfft(y)
        
        return fft_y
    
    def get_bandpass(self):
        ys = 2.0 / self.N * np.abs(self.fft_prep(accel_axis = "z")[1:self.N//2])
        sex = self.sex
        # there should be an algorithm for finding peaks... but let's settle down for now
        peaks, _ = find_peaks(ys, height = [0.02, ])
        peak_hz = peaks * self.nyq / (self.N//2)
        if sex is "M":
            # approximate interval for the fundamental freq for males
            idx = (peak_hz > 80)*(peak_hz < 120)
        else:
            # approximate interval for the fundamental freq for females 
            idx = (peak_hz > 190)*(peak_hz < 230)
        target_freqs = np.where(idx)[0]
        return peak_hz[target_freqs[0]], peak_hz[target_freqs[-1]]
    
    def plot_fft(self, accel_axis = "z", DC_no = True):
        # Why 0 to Nyquist? To avoid aliasing
        xf = np.linspace(0.0, self.nyq, self.N//2)
        yf = self.fft_prep(accel_axis = accel_axis)
        fig, ax = plt.subplots()
        # absolute values mean the amplitude of the fourier transformed data
        # transformed data is symmetric over its center, so half the values are just enough to plot
        if DC_no is not True:
            ax.plot(xf, 2.0 / self.N * np.abs(yf[:self.N//2]))
        else:
            ax.plot(xf[1:], 2.0 / self.N * np.abs(yf[1:self.N//2]))
        ax.set_title("FFT result: %s" % self.name)
        return fig, ax
        
    def psd_prep(self, accel_axis = "z", mode = 1):
        if mode is 1:
            f, Pxx_den = signal.periodogram(getattr(self.df, accel_axis[:]), self.fs)
        elif mode is 2:
            f, Pxx_den = signal.welch(getattr(self.df, accel_axis[:]), self.fs, nperseg = 1024, scaling = "spectrum")
        return f, Pxx_den
    
    def plot_psd(self, accel_axis = "z", mode = 2):
        f, Pxx_den = self.psd_prep(accel_axis = accel_axis, mode = mode)
        fig, ax = plt.subplots()
        ax.semilogy(f, Pxx_den)
        if mode is 1:
            ax.set_title("%s / PSD: Periodogram" % self.name)
        else:
            ax.set_title("%s / PSD: Welch Method" % self.name)
        return fig, ax
    
    def butter_bandpass(self, lowcut = None, highcut = None, order=5):
        if lowcut is None:
            lowcut = self.lowcut
        if highcut is None:
            highcut = self.highcut
        low = lowcut / self.nyq
        high = highcut / self.nyq
        # Scipy bandpass filters designed with b, a are unstable and may result in erroneous filters at high filter orders
        sos = butter(order, [low, high], analog = False, btype='band', output = 'sos')
        return sos
    
    def butter_bandpass_filter(self, lowcut = None, highcut = None, order=5, accel_axis = "z"):
        '''This is numerically erroneous compared to sos'''
        sos = self.butter_bandpass(lowcut, highcut, order=order)
        # sosfilt: filter data along one dimensio using cascaded second-order sections
        y = sosfilt(sos, getattr(self.df, accel_axis[:]))
        return y
    
    '''https://programtalk.com/python-examples/scipy.signal.sosfilt/'''
    def butter_bandpass_sosfiltfilt(self, lowcut = None, highcut = None, accel_axis = "z", order = 5, axis = -1, padtype = "odd", padlen = None, method = 'pad', irlen = None ):
        '''Filtfilt version using Second Order sections. Code is taken from scipy.signal.filtfilt and adapted to make it work with
        sos. Note that broadcasting does not work'''
        data = np.asarray(getattr(self.df, accel_axis[:]))
        sos = self.butter_bandpass(lowcut, highcut, order)

        if padlen is None:
            edge = 0
        else:
            edge = padlen

        if data.shape[axis] <= edge:
            raise ValueError("The length of the input vector x must be at least padlen, which is %d." % edge)

        if padtype is not None and edge > 0:
            if padtype == "even":
                ext = even_ext(data, edge, axis = axis)
            elif padtype == "odd":
                ext = odd_ext(data, edge, axis = axis)
            else:
                ext = const_ext(data, edge, axis = axis)
        else:
            ext = data

        # Get the steady state of the filter's first step resopnse
        zi = sosfilt_zi(sos)

        # Reshape zi and create x0 so that zi*x0 broadcasts to the correct value for the zi keyword argument to lfilter
        x0 = axis_slice(ext, stop = 1, axis = axis)
        # Forward filter
        (y, zf) = sosfilt(sos, ext, axis = axis, zi = zi * x0)

        y0 = axis_slice(y, start = -1, axis = axis)
        # Backward filter
        (y, zf) = sosfilt(sos, axis_reverse(y, axis = axis), axis = axis, zi = zi * y0)
        y = axis_reverse(y, axis = axis)

        if edge > 0:
            y = axis_slice(y, start = edge, stop = -edge, axis = axis)

        return y
    
    def plot_butter(self, lowcut = None, highcut = None, order = 5, accel_axis = "z", mode = 2, env = False):
        if mode is 1:
            y = self.butter_bandpass_filter(lowcut, highcut, order = order, accel_axis = accel_axis)
        else:
            y = self.butter_bandpass_sosfiltfilt(lowcut, highcut, order = order, accel_axis = accel_axis)
        
        t = np.linspace(0, self.N * self.T, self.N, endpoint = False)
        amplitude_envelope = self.get_envelope(y)
        
        fig, ax = plt.subplots()
        ax.plot(t, y, label = "Filtered, %dth order" % order)
        if env:
            ax.plot(t, amplitude_envelope, label = "envelope")
        ax.plot(t, 0.2 * (self.df.trig - 30), '--', label = "Trig")
        ax.set_title("%s / Butterworth filter applied: %s-way" % (self.name, mode))
        ax.set_xlabel("Time in seconds")
        ax.legend(loc = "best")
        return fig, ax
    
    def get_envelope(self, y):
        analytic_signal = hilbert(y)
        amplitude_envelope = np.abs(analytic_signal)
        return amplitude_envelope
    
    def envelope_smoothing(self, amplitude_envelope, window_length):
        # the envelope signal was then modified by applying a moving average filter
        # (implemented as convolution, with a rectangular unit pulse of 40ms in length)
        win = np.repeat([0, 1, 0], window_length)
        convolved = convolve(amplitude_envelope, win, mode = "same") / sum(win)
        return convolved
        
    def plot_envelope(self, lowcut = None, highcut = None, order = 5, accel_axis = "z", mode = 2, window_length = 1500, env = False):
        if mode is 1:
            y = self.butter_bandpass_filter(lowcut, highcut, order = order, accel_axis = accel_axis)
        else:
            y = self.butter_bandpass_sosfiltfilt(lowcut, highcut, order = order, accel_axis = accel_axis)
        
        amplitude_envelope = self.get_envelope(y)
        t = np.linspace(0, self.N * self.T, self.N, endpoint = False)
        
        # plotting begins
        fig, ax = plt.subplots()
        ax.set_xlabel("Time in seconds")
        ax.set_title("%s : Moving Average filter applied" % self.name)
        if env:
            ax.plot(t, amplitude_envelope, label = 'envelope')
        if isinstance(window_length, int):
            convolved = self.envelope_smoothing(amplitude_envelope, window_length)
            ax.plot(t, convolved, label = '%d ms' % window_length)
        else:
            for win_length in window_length:
                convolved = self.envelope_smoothing(amplitude_envelope, win_length)
                ax.plot(t, convolved, linewidth = 0.5, label = '%d ms' % win_length)
        ax.plot(t, (self.df.trig-30)*0.2, '--', label = 'trig')
        ax.legend(loc = "best")
        return fig, ax                

# Actual objects made from the class:accel_data
reuben = accel_data("300mv_Reuben_speech.txt", "M")

# Plotting FFT result
fig, ax = reuben.plot_fft()
ax.set_ylim([0.02, 0.05])

# Plotting the smoothed envelope based on specific cutoffs and window_length
reuben.plot_envelope(lowcut = 80.467, highcut = 119.84, order = 6, window_length = 1600)

# Convert the smoothed envelope to a step function - need further modification
y = reuben.butter_bandpass_sosfiltfilt(reuben.lowcut, reuben.highcut, order = 6, accel_axis = "z")
amplitude_envelope = reuben.get_envelope(y)
convolved = reuben.envelope_smoothing(amplitude_envelope, 1600)

t = np.linspace(0, reuben.N * reuben.T, reuben.N, endpoint = False)
manipulated_convolved = [0.6 if x > 0.357 else 0 for x in convolved]
fig, ax = reuben.plot_envelope(order = 6, mode = 2, env = True, window_length = 1600)
ax.plot(t, np.repeat([0.357], t.shape[0]), label = 'cutoff')
ax.plot(t, manipulated_convolved, '--', linewidth = 2, label = 'vib-on')
ax.legend()
# Run the line below to save the figure
# plt.savefig("%s_MAF_applied.png" % reuben.name, dpi = 300)

# "F" stands for female; females have a different range of fundamental frequency range.
tonya = accel_data("300mv_Tonya_speech.txt", "F")

y = tonya.butter_bandpass_sosfiltfilt(tonya.lowcut, tonya.highcut, order = 6, accel_axis = "z")
amplitude_envelope = tonya.get_envelope(y)
convolved = tonya.envelope_smoothing(amplitude_envelope, 1600)
t = np.linspace(0, tonya.N * tonya.T, tonya.N, endpoint = False)
manipulated_convolved = [0.6 if x > 0.27 else 0 for x in convolved]
fig, ax = tonya.plot_envelope(order = 6, mode = 2, env = True, window_length = 1600)
ax.plot(t, np.repeat([0.27], t.shape[0]), label = 'cutoff')
ax.plot(t, manipulated_convolved, '--', linewidth = 2, label = 'vib-on')
ax.legend()
# Run the line below to save the figre
# plt.savefig("%s_MAF_applied.png" % tonya.name, dpi = 300, figsize = (8, 6)) 
