import sys
import numpy as np
from statistics import pstdev
from sklearn.mixture import GaussianMixture
from scipy.signal import find_peaks, butter, sosfiltfilt, wiener

def findSD1(a):

    # returns the standard deviation of an array, medium method used for spike detection by Quian Quiroga et al

    try:
        a = np.array(a)
    except ValueError:
        print("Warning: Parameter 'a' must be an iterable object.")
        sys.exit("See error message.")


    sd=np.median(np.abs(a))/(0.6745)
    return sd

def findSD2(a):
    # calculate the standard deviation of an array

    return pstdev(np.array(a))

def applyFilter(a, sf, low=500, high=6000):

    nyq = sf / 2
    low = low / nyq
    high = high / nyq
    sos = butter(4, (low,high), btype="pass", output='sos')
    a = sosfiltfilt(sos, a)
    return a

def reduceNoise(a):
    a = wiener(a)
    return a

def func_F(a, t, thre=5, noise=15, sf=40000):
    # thre refers to the  number of standard deviations implemented as the cut-off height for peak finding
    # distance refers to the inter-peak separation, which should be at least the length of the relative refractory period (RRP = 1~2 ms) times the recording frequency.

    distance = int(sf/1000)*2
    d=int((0.5*distance)-1)

    try:
        t = int(t)
    except ValueError:
        print("Warning: Parameter 't' must be an int in seconds representing time of the recording session.")
        sys.exit("See error message.")

    try:
        thre = int(thre)
    except ValueError:
        print("Warning: Parameter 'thre' must be an int representing the lower bound of SD cut-off .")
        sys.exit("See error message.")

    try:
        sf = int(sf)
    except ValueError:
        print("Warning: Parameter 'sf' must be an int in Hertz representing the sampling frequency.")
        sys.exit("See error message.")

    a = applyFilter(a, sf=sf)
    a = reduceNoise(a)
    l_a = len(a)

    p = a.mean()

    SD = findSD1(abs(np.array(a)))
    thre = thre * SD
    height=max(thre,noise)

    active = 1

    i = find_peaks(abs(a), height=height, distance=distance, prominence=p)[0]

    #keep high signals but get rid of artifacts
    l_peaks = a[i]
    if len(l_peaks) >= 1:
        m = abs(np.array(l_peaks)).mean()
        SD_peaks = findSD2(abs(np.array(l_peaks)))
        cut_off = 3 * SD_peaks
        i = np.array([p for p in i if ((m-cut_off) < abs(a[p]) < (m+cut_off))], dtype=np.int64)

    s = i.size
    freq = s / t
    l_peaks = a[i]
    l_peaks = np.abs(l_peaks)

    if s >= 2:
        mixture = GaussianMixture(n_components=2).fit(l_peaks.reshape(-1, 1))
        m = mixture.means_.flatten()
        v = mixture.covariances_.flatten()
        m1 = m[0]
        m2 = m[1]
        sd1 = np.sqrt(v[0])
        sd2 = np.sqrt(v[1])
        if m1+2*sd1 < m2-2*sd2:
            signal = np.array(m)
            signal_var = np.array(v)
        else:
            signal = l_peaks.mean()
            signal = np.array([signal, 0])
            signal_var = l_peaks.var()
            signal_var = np.array([signal_var, 0])
    if s == 1:
        signal = l_peaks.mean()
        signal = np.array([signal, 0])
        signal_var = l_peaks.var()
        signal_var = np.array([signal_var, 0])

    if s >= 1:
        r = []
        for item in i:
            if item-d >= 0 and item+d <= l_a-1:
                r += list(range(item-d, item+d))
            elif item-d < 0:
                r += list(range(item, item+d))
            else:
                r += list(range(item-d, item))

        noise = np.abs(np.delete(a, r)).mean()

    if s == 0 or freq < 1:
        signal = np.array([0,0])
        signal_var = np.array([0,0])
        freq = 0
        active = 0

    SNR = abs(signal[0] / noise)

    if SNR < 1:
        SNR=0
        signal = [0,0]
        signal_var = [0,0]
        freq = 0
        active = 0

    signal_l = signal[0]
    signal_var_l = signal_var[0]
    signal_h = signal[1]
    signal_var_h = signal_var[1]


    l_counter=0
    for i in l_peaks:
        if signal_l-((signal_var_l)**(1/2))*3 < i < signal_l+((signal_var_l)**(1/2))*3:
            l_counter += 1
    freq_l = l_counter/t

    if signal_l > 0 and signal_h > 0:
        ratio = freq_l / freq
    elif signal_l>0 and signal_h==0:
        ratio= 1
    else:
        ratio = None

    return (active, freq, SNR, signal_l, signal_var_l, signal_h, signal_var_h, ratio)
