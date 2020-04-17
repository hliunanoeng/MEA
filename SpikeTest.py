import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from SpikeAnalysis import analyzeByDesign
from scipy.stats import ttest_ind_from_stats as UTtest

def T_table(dfPath,excelPath,filename, skiprow=0):
    # returns a comparison p table to test the statistical significance of the amplitude means measured by different device designs

    df1=analyzeByDesign(dfPath,excelPath,param='Name',skiprow=skiprow)
    rc=df1.index.tolist()
    l_rc=len(rc)

    dummyarray=np.empty((l_rc,l_rc))
    dummyarray[:]=np.nan

    df2=pd.DataFrame(data=dummyarray,index=rc,columns=rc)
    print(df1)
    for r in rc:
        for c in rc:
            if r==c:
                df2.set_value(r,c,np.nan)
            else:
                (mean1, std1, nobs1) = (df1.loc[r]['Amplitude l (uV)'], df1.loc[r]['Amplitude l SD (uV)'], round(df1.loc[r]['Samples (#)'] * df1.loc[r]['Active Percent (%)']))
                (mean2, std2, nobs2) = (df1.loc[c]['Amplitude l (uV)'], df1.loc[c]['Amplitude l SD (uV)'], round(df1.loc[c]['Samples (#)'] * df1.loc[c]['Active Percent (%)']))
                df2.loc[r][c]=round(UTtest(mean1,std1,nobs1,mean2,std2,nobs2,equal_var=False)[1],4)

    path = 'ExcelFiles/' + filename + '.csv'
    df2.to_csv(path, sep='\t', encoding="utf-8")

    return df2

def WilcoxonTest(x,y=None):
    return wilcoxon(x,y,zero_method='wilcox',correction=False)





