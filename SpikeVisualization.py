import sys
import h5py
import pickle
import seaborn as sns
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from scipy import arange
from scipy.signal import find_peaks
from scipy.fftpack import fft
from Util import readH5, readExcel
from SpikeAnalysis import analyzeRecordings
from SpikeDetection import applyFilter, reduceNoise, findSD1, findSD2
from SpikeTest import WilcoxonTest

def plotSpectrum(path,label,fs):
    # plot the signal data in frequency domain.

    try:
        path = str(path)
    except ValueError:
        print("Warning: Parameter 'path' must be a str.")
        sys.exit("See error message.")

    try:
        label = str(label)
    except ValueError:
        print("Warning: Parameter 'label' must be a str. representing the device to be visualized.")
        sys.exit("See error message.")

    (channel, row_labels) = readH5(path)

    f = h5py.File(path, "r", rdcc_nbytes=(1024 ^ 2) * 1000, rdcc_nslots=4)
    stream = f.get("/Data/Recording_0/AnalogStream/Stream_0")
    info = stream.get('InfoChannel')
    #For conversion procedure, please refer to the "HDF5 MCS Raw Data Definition Manual" provided by MultichannelSystems
    exp=float(info[0][7])
    factor=info[0][9]
    V2uV=10**6
    conversion=factor*(10**(exp))*V2uV

    try:
        loc = row_labels.index(label)
    except ValueError:
        print("Warning: Invalid label input. Please note that A01, A16, R01, R16, row J and row Q do not exsit; the rest must be formatted such as B03, E16 etc")
        sys.exit("See error message.")

    c = channel[loc, :]
    c = c*conversion

    n = len(c)  # length of the signal
    k = arange(n)
    T = n / fs
    frq = k / T  # two sides frequency range
    frq = frq[range(int(n / 2))]  # one side frequency range

    Y = fft(c) / n  # fft computing and normalization
    Y = Y[range(int(n / 2))]

    plt.plot(frq, abs(Y), 'r')  # plotting the spectrum
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Y(freq)|')
    plt.show()

    return None

def plotRecordings(request, path, t, thres=5, noise=15, sf=40000, subset=None):
    # plot a feature against device number

    try:
        request = str(request)
    except ValueError:
        print("Warning: Parameter 'request' must be a str.")
        sys.exit("See error message.")

    df = analyzeRecordings(path, t, thres, noise, sf)
    request = request.strip().lower()

    if (not (subset is None)):
        df = df.loc[subset]

    if ('amp' in request):
        x = np.array(df[['amplitude']].index)
        y = np.array(df['amplitude'])
        plt.plot(x, y)
        plt.xlabel("Device Number")
        plt.ylabel("Amplitude Average (uV)")
        plt.show()
        return df['amplitude']
    elif ('freq' in request):
        x = np.array(df[['frequency']].index)
        y = np.array(df['frequency'])
        plt.plot(x, y)
        plt.xlabel("Device Number")
        plt.ylabel("Peak Frequency (Hz)")
        plt.show()
        return df['frequency']
    elif ('snr' in request):
        x = np.array(df[['SNR']].index)
        y = np.array(df['SNR'])
        plt.plot(x, y)
        plt.xlabel("Device Number")
        plt.ylabel("Signal Noise Ratio")
        plt.show()
        return df['SNR']
    else:
        print("Warning: Request type not available.")
        sys.exit("See error message.")

def plotDeviceData(path, label, t,thre=5, noise=15, sf=40000):
    # plot the electrophysiological data of a particular device, peaks included.

    try:
        path = str(path)
    except ValueError:
        print("Warning: Parameter 'path' must be a str.")
        sys.exit("See error message.")

    try:
        label = str(label)
    except ValueError:
        print("Warning: Parameter 'label' must be a str. representing the device to be visualized.")
        sys.exit("See error message.")

    try:
        t = int(t)
    except ValueError:
        print("Warning: Parameter 't' must be an int in seconds representing time of the recording session.")
        sys.exit("See error message.")

    (channel, row_labels) = readH5(path)

    f = h5py.File(path, "r", rdcc_nbytes=(1024 ^ 2) * 1000, rdcc_nslots=4)
    stream = f.get("/Data/Recording_0/AnalogStream/Stream_0")
    info = stream.get('InfoChannel')
    #For conversion procedure, please refer to the "HDF5 MCS Raw Data Definition Manual" provided by MultichannelSystems
    exp=float(info[0][7])
    factor=info[0][10]
    V2uV=10**6
    conversion=factor*(10**(exp))*V2uV

    try:
        loc = row_labels.index(label)
    except ValueError:
        print("Warning: Invalid label input. Please note that A01, A16, R01, R16, row J and row Q do not exsit; the rest must be formatted such as B03, E16 etc")
        sys.exit("See error message.")

    for i in channel:
        np.array(i)

    c = channel[loc, :]
    c = c*conversion
    c = applyFilter(c, sf)
    c = reduceNoise(c)


    SD=findSD1(abs(np.array(c)))
    thre = thre * SD
    height=max(thre,noise)

    p = c.mean()

    peaks, _ = find_peaks(abs(c), height=height, prominence=p, distance=(int(sf/1000)*2))

    #keep high signals but get rid of artifacts
    l_peaks=c[peaks]
    l_peaks_pos=[p for p in l_peaks if p>0]
    l_peaks_neg=[p for p in l_peaks if p<0]
    m_pos=np.array(l_peaks_pos).mean()
    m_neg=np.array(l_peaks_neg).mean()
    if len(l_peaks)>=1:
        SD_peaks_pos=findSD2(l_peaks_pos)
        SD_peaks_neg=findSD2(l_peaks_neg)
        cut_off_pos=3*SD_peaks_pos
        cut_off_neg=3*SD_peaks_neg
        peaks_pos=[p for p in peaks if c[p]>0 and m_pos-cut_off_pos < c[p] < m_pos+cut_off_pos]
        peaks_neg=[p for p in peaks if c[p]<0 and m_neg-cut_off_neg < c[p] < m_neg+cut_off_neg]
        peaks=(peaks_pos+peaks_neg)
        peaks=np.array(peaks)

    time = np.linspace(0, t, num=len(c))
    plt.rcParams['agg.path.chunksize']=10000
    plt.plot(time, c, 'b', alpha=0.5)
    if len(peaks)>=1:
        plt.plot(peaks / (len(c) / t), c[peaks], "rx")
    plt.xlabel("Time (s)")
    plt.ylabel("Action Potential (uV)")
    plt.title(label)
    plt.show()

def violinByDesign(request, dfPath, excelPath, param, skiprow=1,exclude=[],title=''):

    # violin plot by device based on amplitude, frequency or SNR

    try:
        request = str(request)
    except ValueError:
        print("Warning: Parameter 'request' must be a str.")
        sys.exit("See error message.")

    try:
        param = str(param)
    except ValueError:
        print("Warning: Parameter 'request' must be a str.")
        sys.exit("See error message.")

    with open(dfPath, "rb") as pickle_in:
        df1 = pickle.load(pickle_in)
        dic = readExcel(excelPath, param, skiprow=skiprow)
        adjustment=len(list(dic.keys()))
        if not (exclude == []):
            df1.drop(exclude,inplace=True)
            for i in exclude:
                for j in dic.values():
                    if i in j:
                        j.remove(i)
                    continue

        for i in dic.keys():
            S=i.split(' ')[1]
            if len(S)<3:
                s=i.split(' ')[0]+' '+'S0'+S[1]+' '+i.split(' ')[2]+' '+i.split(' ')[3]
                dic[s]=dic.pop(i)

        request = request.strip().lower()
        df2 = df1.set_index([list(range(len(df1.index)))])
        df2=df2.assign(Design=None,Device=df1.index)

        df2=df2.copy().loc[df2['Active']==1]
        l_df=len(df2)
        df2.reset_index(inplace=True)
        df2.drop(['index'],axis=1,inplace=True)

        sns.set(style="whitegrid")

        for k in dic.keys():
            for i in range(l_df):
                device = df2.loc[i, 'Device']
                if (device in dic[k]):
                    df2.loc[i,'Design'] = k

        xlabel = param.capitalize()
        if xlabel in "Openings":
            xlabel='Openings (#)'
        elif xlabel in 'Width':
            xlabel='Width (um)'
        elif xlabel in 'Diameter':
            xlabel='Diameter (um)'
        elif xlabel in 'Name':
            xlabel='Electrode Design'
        else:
            print('Input design parameter not recognized')
            sys.exit('See error message.')

        l_order=list(set(list(df2.Design)))
        if None in l_order:
            l_order.remove(None)
        l_order=sorted(l_order)

        if ('amp' in request):
            ax = sns.violinplot(x=df2.Design,y=df2.Amplitude_l, order=l_order, cut=0, inner='box', scale='count',notch=True)
            if adjustment>10:
                plt.xticks(rotation=60,size=5)
            ax.set_xlabel('\n '+xlabel)
            ax.set_ylabel('Amplitude ($\mu$V)\n ')
            plt.xticks(size=5)
            plt.yticks(size=5)
            plt.grid()
            plt.title(title)
            plt.show()
        elif ('freq' in request):
            ax = sns.violinplot(x=df2.Design,y=df2.Frequency,order=l_order,cut=0, inner='box', scale='count')
            if adjustment>10:
                plt.xticks(rotation=60,size=5)
            ax.set_xlabel('\n '+xlabel)
            ax.set_ylabel('Spike Rate (Hz)\n ')
            plt.xticks(size=5)
            plt.yticks(size=5)
            plt.grid()
            plt.title(title)
            plt.show()
        elif ('snr' in request):
            ax = sns.violinplot(x=df2.Design,y=df2.SNR, order=l_order,cut=0,inner='box', scale='count')
            if adjustment>15:
                plt.xticks(rotation=60,size=5)
            ax.set_xlabel('\n '+xlabel)
            ax.set_ylabel('Signal Noise Ratio\n ')
            plt.xticks(size=5)
            plt.yticks(size=5)
            plt.grid()
            plt.title(title)
            plt.show()
        # print(df2)

def slopePlot(l_set,dic):
    style.use('ggplot')
    l_a=l_set[0]

    for i in l_set:
        if len(l_a)!=len(i):
            raise Exception("Two iterables l1 and l2 must be of the same length.")

    l=len(l_set)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    for i in range(l-1):
        l_a=l_set[i]
        l_b=l_set[i+1]
        for j in range(len(l_a)):
            if l_a[j]<l_b[j]:
                s="k-"
            elif l_b[j]<l_a[j]:
                s="k-"
            else:
                s='gray'

            ax1.plot([i+1,i+2],[l_a[j],l_b[j]],s,alpha=0.35)

        ax1.scatter([i+1]*len(l_a),l_a,c='k',s=10)
        ax1.scatter([i+2]*len(l_b),l_b,c='k',s=10)

        ax1.text(i+1.5,min(l_a+l_b)+((max(l_a+l_b)-min(l_a+l_b))/2),"p="+str(WilcoxonTest(l_a,l_b)[1])[:6],size=15)

        #ax1.boxplot([l_a,l_b],positions=[i+1,i+2])

    l_together=[]
    for i in l_set:
        l_together+=i
    ybounds=np.linspace(min(l_together)-1,max(l_together)+1,num=5,endpoint=True)

    xticklabels=[]
    for i in range(1,l+1):
        str_base="label"
        xticklabels.append(dic[str_base+str(i)])

    ax1.set_xbound(0,l+1)
    ax1.set_xticks(list(range(1,l+1)))
    ax1.set_xticklabels(xticklabels,size=12)
    ax1.set_xlabel(dic['xlabel'],size=12)
    # ax1.set_ybound(ybounds[0],ybounds[-1]+1)
    # ax1.set_yticks(ybounds+1)
    # ax1.set_yticklabels(ybounds,size=12)
    ax1.set_ybound(-10,100)
    ax1.set_yticks([-10,0,20,40,60,80,100])
    ax1.set_yticklabels(['',0,20,40,60,80,100])

    ax1.set_ylabel(dic['ylabel'],size=15)

    plt.title(dic['title'],size=12)
    plt.show()

#Diameter Activity Relationship
#slopePlot([[0,0,0,0,0,0],[10,0,0,0,0,10],[0,10,0,0,12,10],[50,28,61,28,35,56]],{'label1':'45 $\mu$m','label2':'60 $\mu$m','label3':'90 $\mu$m','label4':'150 $\mu$m','xlabel':'Igloo Diameter ($\mu$m)','ylabel':'Activity (%)','title':"Diameter & Activity Relationship"})
#slopePlot([[0,0,0,0,0,0,61,10,10,0,28,50],[60,65,80,98,95,92,90,99,99,96,90,86]],{'label1':'Smaller Diameter, Fewer Openings','label2':'Larger Diameter, More Openings','xlabel':'','ylabel':'Activity (%)', 'title':''})

#Culture & Days In Vitro

'''
D2=[30,100,40,100,75,100,55,100,25,40,52,78,100,100,100,100,42,25,40,90,95,100,100,100,20,55,60,72,100,100,100,100,28,55,52,54,100,100,100,95,10,50,54,80,100,100,100,100]
D3=[0,0,0,10,0,0,10,50,0,0,0,8,0,0,0,28,0,0,0,0,0,0,10,28,0,0,10,8,0,0,10,28,0,0,0,8,0,0,0,61,0,0,0,0,0,10,10,61]

DIV5=[30,40,75,55,0,0,0,10,25,40,52,78,0,0,0,8,42,25,40,90,0,0,0,0,20,55,60,72,0,0,10,8,28,55,52,54,0,0,0,8,0,0,0,0,10,50,54,80]
DIV7=[100,100,100,100,0,10,0,50,100,100,100,100,0,0,0,28,95,100,100,100,0,0,10,28,100,100,100,100,0,0,10,28,100,100,100,95,0,0,0,61,0,10,10,61,100,100,100,100]

D=[]
for i in range(len(D2)):
    D.append(D2[i])
    D.append(D3[i])

DIV=[]
for i in range(len(DIV5)):
    DIV.append(DIV5[i])
    DIV.append(DIV7[i])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot([1,2]*len(D2), D, "k-", alpha=0.35)
ax1.plot([3,4]*len(DIV5), DIV, "k-", alpha=0.35)
ax1.scatter([1] * len(D2) + [2] * len(D3) + [3] * len(DIV5) + [4] * len(DIV7), D2+D3+DIV5+DIV7, c='k', s=10)
#ax1.boxplot([D2, D3, DIV5, DIV7], positions=[1, 2, 3, 4])

print(WilcoxonTest(D2,D3))
print(WilcoxonTest(DIV5,DIV7))

ax1.set_xticks(list([1,2,3,4]))
ax1.set_xticklabels(['2D','3D','5DIV','7DIV'], size=12)
ax1.set_ybound(-10, 100)
ax1.set_yticks([-10,0,20,40,60,80,100])
ax1.set_yticklabels(['',0,20,40,60,80,100])
ax1.set_ylabel('Activity (%)', size=15)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.show()

'''

'''
Diameter45=[0,0,0,0,0,0,0,0,0,0,0,0]
Diameter60=[0,10,0,0,0,0,0,0,0,0,0,10]
Diameter90=[0,0,0,0,0,10,10,10,0,0,0,10]
Diameter150=[10,50,10,28,0,25,10,30,10,61,0,60]

fig = plt.figure()
ax1 = fig.add_subplot(111)

for i in range(len(Diameter45)):
    ax1.plot([1,2], [Diameter45[i],Diameter60[i]], "k-", alpha=0.35)
    ax1.plot([2,3], [Diameter60[i],Diameter90[i]], "k-", alpha=0.35)
    ax1.plot([3,4], [Diameter90[i],Diameter150[i]], "k-", alpha=0.35)

print(WilcoxonTest(Diameter45,Diameter60))
print(WilcoxonTest(Diameter60,Diameter90))
print(WilcoxonTest(Diameter90,Diameter150))

Opening2=[0,10,0,50,0,0,0,25]
Opening3=[0,0,10,25,0,0,10,25]
Opening6=[0,0,0,61,0,10,10,60]

for i in range(len(Opening2)):
    ax1.plot([5,6], [Opening2[i],Opening3[i]], "k-", alpha=0.35)
    ax1.plot([6,7], [Opening3[i],Opening6[i]], "k-", alpha=0.35)

print(WilcoxonTest(Opening2,Opening3))
print(WilcoxonTest(Opening3,Opening6))

Width5=[0,10,0,50,0,0,8,35,0,0,0,61,0,0,0,60,0,0,0,0,0,0,5]
Width7=[0,0,0,30,0,0,8,30,0,10,10,60,0,0,0,6,0,0,0,35,0,10,100]

for i in range(len(Width5)):
    ax1.plot([8,9], [Width5[i],Width7[i]], "k-", alpha=0.35)

print(WilcoxonTest(Width5,Width7))


ax1.scatter([1] * len(Diameter45) + [2] * len(Diameter60) + [3] * len(Diameter90) + [4] * len(Diameter150), Diameter45+Diameter60+Diameter90+Diameter150, c='k', s=10)

ax1.scatter([5] * len(Opening2) + [6] * len(Opening3) + [7] * len(Opening6), Opening2+Opening3+Opening6, c='k', s=10)

ax1.scatter([8] * len(Width5) + [9] * len(Width7), Width5+Width7, c='k', s=10)

#ax1.boxplot([Diameter45, Diameter60, Diameter90, Diameter150,Opening2, Opening3, Opening6,Width5,Width7], positions=[1, 2, 3, 4,5,6,7,8,9])

ax1.set_xticks(list([1,2,3,4,5,6,7,8,9]))
ax1.set_xticklabels(['Diameter\n45$\mu$m','Diameter\n60$\mu$m','Diameter\n90$\mu$m','Diameter\n150$\mu$m','Opening 2','Opening 3','Opening 6','Width\n5$\mu$m','Width\n7.5$\mu$m'], size=12)
ax1.set_ybound(-10, 100)
ax1.set_yticks([-10,0,20,40,60,80,100])
ax1.set_yticklabels(['',0,20,40,60,80,100])
ax1.set_ylabel('Activity (%)', size=15)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.show()

'''
