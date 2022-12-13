import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

df = pd.read_csv('ndvi_2017.csv')
df.keys()

# Checking number of fields in the csv file (It should be 80) 
df.set_index('ID', inplace= True)
df[df.index.duplicated()]
df_c = df[~df.index.duplicated()]
len(df_c)  #80

field_select = df_c.index.tolist()

# Checing number of sentinal 2 images 
df.set_index('date', inplace= True)
df[df.index.duplicated()]
df_c = df[~df.index.duplicated()]
len(df_c) # 40

len(df)

####################################################################

# creating list of datafreames for specific fields
df = pd.read_csv('ndvi_2017.csv')

df.head()
df['datetime']= pd.to_datetime(df['date'])
df.drop(columns = 'date', inplace = True)
df.set_index('datetime', inplace = True)
df.keys()

dfs = []
for i in field_select:
    df_s = df[df['ID'] == i].copy()
    dfs.append(df_s)

print(dfs[1])


def data_minning(df):
    # Step 3.1 Adjust data that looks suspect
    df_sam = pd.DataFrame()
    df_sam[['CropTyp','ID','VI']] = df[['CropTyp','ID','nd']].copy()
    x = df_sam[df_sam.index.duplicated()]
    print(f'There are {x.count()} duplicate days')
    df_sam = df_sam[~df_sam.index.duplicated()]
    df_sam['DOY'] = [df_sam.index[i].dayofyear for i in range(len(df_sam))]
    # Step 3.1.1 Removing NA values, replacing negative values to 0 and > 1 to 1
    df_sam.dropna(inplace = True)

    df_sam[df_sam['VI'] < 0] = 0

    df_sam[df_sam['VI'] > 1] = 1

    # Step 3.1.2 Adjust unexpected jumps 
    df_sam['diff VI'] = df_sam['VI'].diff()
    df_sam['diff DOY'] = df_sam['DOY'].diff()
    df_sam['jmp_th'] = df_sam['diff VI'] / df_sam['diff DOY'] 
    df_sam.dropna(inplace=True)
    df_sam['VI'].mask(df_sam['jmp_th'] > 0.015, np.nan, inplace=True)
    unex_jmp = np.where(df_sam['jmp_th'] > 0.015)
    print(f'There is unexpected jumps in {unex_jmp}')
    df_sam['VI'].interpolate(method='linear', inplace= True) #linear interpolation

    # Step 3.2 Regularize (create equidistant data with one value for every 10 days) 
    df_sam = df_sam.resample('10D').max()
    df_sam.dropna(inplace=True)

    # Step 3.3 Smooth time series using the Savitzky- Golay R package  
    df_sam['VI_smoothened'] = savgol_filter(df_sam['VI'], window_length=5, polyorder=1, axis=0)
    df_sam.head()
    


    # Step 4 Linearly interpolate to get a daily time series 
    df_out = pd.DataFrame()
    df_out['VI_smoothened'] = df_sam['VI_smoothened'].copy()
    df_out = df_out.resample('D').asfreq()
    df_out['VI_smoothened'].interpolate(method='linear', inplace= True)
    return df_out, df_sam


df_VI_cleaned = []
df_VI_ref = []
for field in dfs:
    df_out, df_sam = data_minning(field)
    df_VI_cleaned.append(df_out)
    df_VI_ref.append(df_sam)


df_VI = pd.concat(df_VI_cleaned,axis=1)
df_VI.head()
df_VI.to_csv('NDVI_daily_time_series.csv')

df_ref = pd.concat(df_VI_ref)
df_ref.head()
df_ref.set_index('ID',inplace=True)

df_feature = df_ref[~df_ref.index.duplicated()]
df_feature.head()
df_feature['CropTyp'].to_csv('Field_ID_crop_type.csv')