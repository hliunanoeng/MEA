import h5py
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from pathlib import Path
import os
import sys

pd.set_option('display.max_rows', 300, 'display.max_columns', 10)  # pypass autosizing summary view
# print(style.available)

def evenSpacedString(s):
    l = s.split(' ')
    st = str()
    for i in l:
        if i != '':
            st = st + i + ' '
    st=st.strip()
    return st

def createTestFile(path):

    # creates a test file in the directory where the original file is located.

    try:
        path = str(path)
    except ValueError:
        print("Warning: Parameter 'path' must be a str.")
        sys.exit("See error message.")

    try:
        f1 = h5py.File(path, "r")
    except FileNotFoundError:
        print("Warning: Cannot open the file, please check the file path.")
        sys.exit("See error message.")

    l = path.split(".")
    path2 = l[0] + '_test.' + l[1]

    my_file = Path(path2)

    if my_file.is_file():
        print("Warning: File already exists, please delete and re-run.")
        os.remove(path2)
        sys.exit("See error message.")

    else:
        f2 = h5py.File(path2, "a")
        g = f2.create_group('/Data/Recording_0/AnalogStream')
        f1.copy('/Data/Recording_0/AnalogStream/Stream_0', g)

    return path2

def selectTimeSequence(path,end,start=0,step=1,freq=40000):

    # down-sample and/or slice the recording data.

    if end <= start:
        print("Warning: the end time must be great than the start time.")
        sys.exit("See error message.")

    path2 = createTestFile(path)

    f2 = h5py.File(path2, "a")
    Stream_0 = f2.get('/Data/Recording_0/AnalogStream/Stream_0')
    ChannelData = Stream_0['ChannelData']

    c = ChannelData[:, (start*freq):(end*freq):step]
    Stream_0.__delitem__('ChannelData')
    Stream_0['ChannelData'] = c

    return path2

def readH5(path):

    # read an h5 file and return channel data along with device labels

    try:
        path = str(path)
    except ValueError:
        print("Warning: Parameter 'path' must be a str.")
        sys.exit("See error message.")

    try:
        f = h5py.File(path, "r", rdcc_nbytes=(1024^2)*1000, rdcc_nslots=4)
    except FileNotFoundError:
        print("Warning: Cannot open the file, please check the file path.")
        sys.exit("See error message.")

    stream = f.get("/Data/Recording_0/AnalogStream/Stream_0")
    channelData = stream.get('ChannelData')
    info = stream.get('InfoChannel')

    # for i in info:
    #     print(i)

    # row_labels = [chr(65 + i[2])+i[3].decode('UTF-8') for i in info]
    row_labels = [chr(65 + i[2])+i[4].decode('UTF-8') for i in info]
    row_labels = [i[1:] for i in row_labels]

    for i in range(len(row_labels)):
        if len(row_labels[i]) < 3:
            row_labels[i] = row_labels[i][0] + '0' + row_labels[i][1]

    return (channelData, row_labels)

def readExcel(path,param,skiprow):

    # read an Excel file and return a dictionary of devices group by design

    try:
        path = str(path)
    except ValueError:
        print("Warning: Parameter 'path' must be a str.")
        sys.exit("See error message.")

    try:
        df_main = pd.read_excel(path, sheet_name="Sheet1", skiprows=skiprow)
    except FileNotFoundError:
        print("Warning: Cannot open the file, please check the file path.")
        sys.exit("See error message.")

    columns=list(df_main.columns.values)
    new_columns=[i.split('(')[0].strip() for i in columns]
    new_columns=[i.lower() for i in new_columns]

    l_check=[]
    for i in new_columns:
        l_check.append(param.split(' ')[0].lower() == i)

    if not any(l_check):
        print("The input request is invalid.")
        sys.exit("See error message.")

    header=dict()
    for i in range(len(new_columns)):
        header[columns[i]]=new_columns[i]

    df_main.rename(columns=header,inplace=True)
    df_main.fillna(0,inplace=True)

    df_main['name']=df_main['name'].apply(lambda x: evenSpacedString(x))

    param = param.split(' ')[0].lower()

    s = set(df_main[param])

    dic = dict()
    for i in s:
        dic[i] = []

    df_sub = df_main[[param, 'electrode id']].copy()

    l1 = [i.strip().upper() for i in df_sub['electrode id']]
    l2 = []
    for i in l1:
        if len(i) < 3:
            i = i[0] + '0' + i[1]
        l2.append(i)

    df_sub['electrode id'] = l2
    for i in range(len(df_sub[param])):
        dic[df_sub[param][i]].append(df_sub['electrode id'][i])

    return dic



