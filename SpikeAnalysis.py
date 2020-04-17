import sys
import pickle
import h5py
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import repeat
from pathlib import Path
from Util import readExcel, readH5
from SpikeDetection import func_F

def analyzeRecordings(path, t, thre=5, noise=15, sf=40000):
    # creates a pickle + csv and returns a dataframe of relevant features

    df_file = Path("DataframePickle/df.pickle")

    if df_file.is_file():
        print("Warning: File already exists, please delete or rename then run again.")
        sys.exit("See error message.")

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

    (channel, row_labels) = readH5(path)

    #For conversion procedure, please refer to the "HDF5 MCS Raw Data Definition Manual" provided by MultichannelSystems

    f = h5py.File(path, "r", rdcc_nbytes=(1024 ^ 2) * 1000, rdcc_nslots=4)
    stream = f.get("/Data/Recording_0/AnalogStream/Stream_0")
    info = stream.get('InfoChannel')
    # exp = float(info[0][6])
    exp = float(info[0][7])
    # factor = info[0][9]
    factor = info[0][10]
    V2uV = 10**6
    conversion = factor*(10**exp)*V2uV

    num_workers = mp.cpu_count()

    pool = mp.Pool(processes=(num_workers-1))
    l = pool.starmap(func_F, zip(channel, repeat(t), repeat(thre), repeat(noise), repeat(sf)))
    pool.close()
    pool.join()

    active = np.array([i[0] for i in l])
    frequency = np.array([i[1] for i in l])
    SNR = np.array([i[2] for i in l])
    amplitude1 = np.array([i[3] for i in l])*conversion
    amplitude_var1 = np.array([i[4] for i in l])*(conversion**2)
    amplitude2 = np.array([i[5] for i in l])*conversion
    amplitude_var2 = np.array([i[6] for i in l])*(conversion**2)
    ratio = np.array([i[7] for i in l])


    df = {'Active': active,'Frequency': frequency,'SNR': SNR, 'Amplitude_l': amplitude1, 'AmplitudeVar_l': amplitude_var1,'Amplitude_h': amplitude2, 'AmplitudeVar_h': amplitude_var2, 'Ratio': ratio}
    df = pd.DataFrame(data=df, index=row_labels)
    df.sort_index(inplace=True)

    with open("DataframePickle/df.pickle", "wb") as pickle_out:
        pickle.dump(df, pickle_out)

    df_copy = df.copy()
    df_copy['AmplitudeVar_l'] = df_copy['AmplitudeVar_l'].apply(lambda x: round(x**0.5, 2))
    df_copy['Amplitude_l'] = df_copy['Amplitude_l'].apply(lambda x: round(x, 2))
    df_copy['AmplitudeVar_h'] = df_copy['AmplitudeVar_h'].apply(lambda x: round(x**0.5, 2))
    df_copy['Amplitude_h'] = df_copy['Amplitude_h'].apply(lambda x: round(x, 2))
    df_copy['Ratio'] = df_copy['Ratio'].apply(lambda x: round(x*100, 4) if (x is not None) else None)
    df_copy=df_copy.round({'Active': 0,'SNR': 2, "Frequency": 2})
    df_copy.rename(columns={'Active':'Active','Frequency':'Frequency (Hz)', 'SNR':'SNR','Amplitude_l':'Amplitude l (uV)','AmplitudeVar_l':'Amplitude SD l (uV)','Amplitude_h':'Amplitude h (uV)','AmplitudeVar_h':'Amplitude SD h (uV)', 'Ratio':'Ratio (%)'}, inplace=True)
    df_copy.to_csv('ExcelFiles/df.csv',sep='\t',encoding="utf-8")

    return df

def analyzeByDesign(dfPath, excelPath, param, skiprow=1,exclude=[]):

    # group the results from analyzeRecordings by device design and create outputs in csv format
    # dfPath: path to the pickled dataframe analyzed by analyzeRecordings
    # excelPath: path to the excel design sheet

    with open(dfPath, "rb") as pickle_in:
        df = pickle.load(pickle_in)
        dic = readExcel(excelPath, param,skiprow=skiprow)
        if not (exclude == []):
            df.drop(exclude,inplace=True)
            for i in exclude:
                for j in dic.values():
                    if i in j:
                        j.remove(i)
                    continue


        columns = list(df.columns.values)
        columns[0]="Active Percent"
        columns[4]='Amplitude l SD'
        columns[6]='Amplitude h SD'
        columns.append('Samples')
        columns.append('SNR SD')
        columns.append('Frequency SD')
        dic2 = dict()

        for i in list(dic.keys()):
            df_temp=df.loc[dic[i]]
            num=0
            total=len(dic[i])

            for j in list(df_temp['Active']):
                if j==0:
                    num+=1

            active_percentage=((total-num)/total)

            df_temp2 = df_temp.copy()
            df_temp2=df_temp2.replace(0,np.NAN)
            df_temp2=df_temp2[['SNR','Frequency']].std(skipna=True)


            df_temp3=df_temp.copy()
            df_temp3=df_temp3.replace(np.NAN,0)
            df_temp3['Weight_l']=df_temp3['Frequency']*df_temp3['Ratio']
            df_temp3['Weight_h']=df_temp3['Frequency']-df_temp3['Frequency']*df_temp3['Ratio']

            Amplitude_l_num=(df_temp3['Amplitude_l']*df_temp3['Weight_l']).sum()
            Amplitude_l_denom=(df_temp3['Weight_l'].sum())
            if Amplitude_l_denom==0:
                Amplitude_l=0
            else:
                Amplitude_l=Amplitude_l_num/Amplitude_l_denom

            Amplitude_h_num=(df_temp3['Amplitude_h']*df_temp3['Weight_h']).sum()
            Amplitude_h_denom=(df_temp3['Weight_h'].sum())
            if Amplitude_h_denom==0:
                Amplitude_h=0
            else:
                Amplitude_h=Amplitude_h_num/Amplitude_h_denom

            df_temp3['Amplitude_l_sqrt']=df_temp3['Amplitude_l'].apply(lambda x: x**2)
            df_temp3['Amplitude_h_sqrt']=df_temp3['Amplitude_h'].apply(lambda x: x**2)

            Amplitude_l_SD_num=((df_temp3['Amplitude_l_sqrt']+df_temp3['AmplitudeVar_l'])*df_temp3['Weight_l']).sum()
            if Amplitude_l_denom==0:
                Amplitude_l_SD=np.NAN
            else:
                Amplitude_l_SD=((Amplitude_l_SD_num/Amplitude_l_denom)-(Amplitude_l**2))**(1/2)

            Amplitude_h_SD_num=((df_temp3['Amplitude_h_sqrt']+df_temp3['AmplitudeVar_h'])*df_temp3['Weight_h']).sum()
            if Amplitude_h_denom==0:
                Amplitude_h_SD=np.NAN
            else:
                Amplitude_h_SD=((Amplitude_h_SD_num/Amplitude_h_denom)-(Amplitude_h**2))**(1/2)

            dic_temp=dict()
            dic_temp["SNR SD"]=df_temp2['SNR']
            dic_temp['Frequency SD']=df_temp2['Frequency']
            dic_temp['Active Percent'] = active_percentage
            dic_temp['Samples'] = total
            dic_temp['Amplitude l']=float(Amplitude_l)
            dic_temp['Amplitude h']=float(Amplitude_h)

            if Amplitude_l_SD:
                dic_temp['Amplitude l SD']=float(Amplitude_l_SD)
            else:
                dic_temp['Amplitude l SD'] = np.NAN

            if Amplitude_h_SD:
                dic_temp['Amplitude h SD']=float(Amplitude_h_SD)
            else:
                dic_temp['Amplitude h SD'] = np.NAN

            # df_temp = df_temp.replace(0,np.NAN)
            l_temp= list(df_temp.mean(skipna=True))

            l_temp.pop(0)
            l_temp.insert(0,dic_temp["Active Percent"])
            l_temp.pop(3)
            l_temp.insert(3,dic_temp["Amplitude l"])
            l_temp.pop(4)
            l_temp.insert(4,dic_temp["Amplitude l SD"])
            l_temp.pop(5)
            l_temp.insert(5,dic_temp["Amplitude h"])
            l_temp.pop(6)
            l_temp.insert(6,dic_temp["Amplitude h SD"])
            l_temp.append(dic_temp['Samples'])
            l_temp.append(dic_temp["SNR SD"])
            l_temp.append(dic_temp["Frequency SD"])
            dic2[i] = l_temp

        l = []
        for i in dic2.values():
            l.append(i)

        index = []
        for i in dic2.keys():
            index.append(i)

        pd2 = pd.DataFrame(data=l,columns=columns,index=index)
        pd2.rename(
            columns={'Active Percent':"Active Percent (%)",'Samples': 'Samples (#)','Amplitude_l': 'Amplitude l (uV)', 'Amplitude l SD': 'Amplitude l SD (uV)','Amplitude_h':'Amplitude h (uV)','Amplitude h SD':'Amplitude h SD (uV)', 'Frequency': 'Frequency (Hz)','Frequency SD':'Frequency SD (Hz)'},
            inplace=True)

        pd2['Active Percent (%)']=pd2['Active Percent (%)'].apply(lambda x: x*100)
        pd2=pd2[['Samples (#)','Active Percent (%)','SNR','SNR SD','Frequency (Hz)','Frequency SD (Hz)','Amplitude l (uV)','Amplitude l SD (uV)','Amplitude h (uV)', 'Amplitude h SD (uV)']]
        pd2 = pd2.round(2)
        pd2.sort_index(inplace=True)

        path='ExcelFiles/'+'df_By'+param.capitalize()+'.csv'
        pd2.to_csv(path, sep='\t', encoding="utf-8")

    return pd2





