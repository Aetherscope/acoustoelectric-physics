# ///////////////////////////////////////////////////////////////
#
# Measurements class for Acoustoelectric measurements.
# Copyright (C) 2024 Christopher Chare
#
# License: CC BY-NC-SA 4.0
#
# ///////////////////////////////////////////////////////////////
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import datetime
import numpy as np
import pandas as pd
from enum import Enum

COLUMNS = [
    'X','Y','Z',
    'Amplitude X','Amplitude Y',
    'Phase X','Phase Y'
]
PARAMS = [
    "Measurement Date", "Measurement Description",
    "Sampling Rate",
    "Carrier Frequency", "Injection Frequency",
    "Channel Gain"
]

class Measurements():

    ''' Class for measurement acquisition '''
    def __init__(self): 
        self.data = pd.DataFrame(columns=COLUMNS)

    def initializeDataFrame(self,xArray,yArray,zArray):
        del self.data
        permutedArray = np.stack(np.meshgrid(xArray,yArray,zArray),-1).reshape(-1,3)
        initialDataFrameMatrix = np.zeros((permutedArray.shape[0],len(COLUMNS)))
        initialDataFrameMatrix[:permutedArray.shape[0],:permutedArray.shape[1]] = permutedArray
        self.data = pd.DataFrame(initialDataFrameMatrix,columns=COLUMNS)

    def initializeParameters(self,description):
        self.parameter = dict.fromkeys(PARAMS)
        self.parameter["Measurement Date"] = datetime.datetime.now()
        self.parameter["Measurement Description"] = description
        self.parameter["Sampling Rate"] = 0
        self.parameter["Carrier Frequency"] = 985000
        self.parameter["Injection Frequency"] = 1000
        self.parameter["Channel Gain"] = {'X':1000,'Y':1000}

    def updateInstance(self,coords,data,Xon=True,Yon=True,burst=False):
        index = self.data[
            (self.data['X']==coords['X']) &
            (self.data['Y']==coords['Y']) &
            (self.data['Z']==coords['Z'])].index.tolist()[0]
        print(f'Measurement index: {index}')
        print(f'Measurement coordinates: {coords}')
        amp_X, amp_Y, ph_X, ph_Y = 0, 0, 0, 0
        if Xon:
            amp_X, ph_X = self._parseFFT(data['Demod X'],self.parameter["Injection Frequency"],self.parameter["Channel Gain"]['X'],self.parameter["Sampling Rate"])
            if burst:
                amp_X = (np.max(data['Demod X'])-np.min(data['Demod X']))/2
            self.data.at[index,'Amplitude X'] = amp_X
            self.data.at[index,'Phase X'] = ph_X
        if Yon:
            amp_Y, ph_Y = self._parseFFT(data['Demod Y'],self.parameter["Injection Frequency"],self.parameter["Channel Gain"]['Y'],self.parameter["Sampling Rate"])
            if burst:
                amp_Y = (np.max(data['Demod Y'])-np.min(data['Demod Y']))/2
            self.data.at[index,'Amplitude Y'] = amp_Y
            self.data.at[index,'Phase Y'] = ph_Y
        print(f'Measurement instance: {amp_X}:{amp_Y}:{ph_X}:{ph_Y}')
        return amp_X #max(amp_X,amp_Y)
    
    def maskDeadzone(self,deadzones,axes,resolution):
        skip_pts = 0
        for idx in range(len(self.data.index)):
            for zone in deadzones:
                if (((self.data.iloc[idx][axes[0]] >= np.min(zone['X'])-resolution/2) and (self.data.iloc[idx][axes[0]] <= np.max(zone['X'])+resolution/2)) and
                    ((self.data.iloc[idx][axes[1]] >= np.min(zone['Y'])-resolution/2) and (self.data.iloc[idx][axes[1]] <= np.max(zone['Y'])+resolution/2))):
                    self.data.at[idx,'Amplitude X'] = np.nan
                    self.data.at[idx,'Amplitude Y'] = np.nan
                    self.data.at[idx,'Phase X'] = np.nan
                    self.data.at[idx,'Phase Y'] = np.nan
                    skip_pts += 1
        return skip_pts

    def findIndex(self,xPos,yPos,zPos):
        idx_X = self.data['X']==xPos
        idx_Y = self.data['Y']==yPos
        idx_Z = self.data['Z']==zPos
        return np.where(idx_X & idx_Y & idx_Z)[0]
    
    def findMax(self):
        return self.data.loc[self.data['Amplitude Y'].idxmax()]

    def fetchInstance(self,columns,xPos=None,yPos=None,zPos=None,index=None): #columns=['X','Y','Z'] etc.
        if index is None:
            index = self.data[
                (self.data['X']==xPos) &
                (self.data['Y']==yPos) &
                (self.data['Z']==zPos)].index.tolist()[0]
        return self.data.iloc[index][columns]

    def fetchMatrix(self,primary,coords,component='X'):
        df = self.data.loc[(self.data[coords[0]]==primary)]
        return df.pivot(index=coords[1],columns=coords[2],values=f'Amplitude {component}').values,df.pivot(index=coords[1],columns=coords[2],values=f'Phase {component}').values

    def randomData(self):
        idx_count = self.data.shape[0]
        temp_array = np.sin(np.linspace(0,4*np.pi,idx_count))
        self.data['Amplitude X'] = np.abs(temp_array)
        self.data['Phase X'] = temp_array*np.pi
        temp_array = np.sin(np.linspace(0,2*np.pi,idx_count))
        self.data['Amplitude Y'] = np.abs(temp_array)
        self.data['Phase Y'] = temp_array*np.pi

    def _find_nearest_bin(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def _parseFFT(self,data,frequency,gain,smplrate):
        np_fft_val = np.fft.fft(data)
        np_fft_freq = np.fft.fftfreq(len(data), d = 1/smplrate)
        np_fft_maxVal_idx = self._find_nearest_bin(np_fft_freq,frequency)
        
        amplitude = (2 / len(data) * np.abs(np_fft_val[np_fft_maxVal_idx])) 
        phase = np.angle(np_fft_val[np_fft_maxVal_idx])

        return amplitude/gain, phase

COLUMNS_SWEEP = [
    'Pressure','Dipole',
    'Amplitude X','Amplitude Y',
    'Phase X','Phase Y'
]

class Sweep():
    def __init__(self):
        self.data = pd.DataFrame(columns=COLUMNS_SWEEP)

    def initializeDataFrame(self,pArray,dArray):
        del self.data
        permutedArray = np.stack(np.meshgrid(pArray,dArray),-1).reshape(-1,2)
        initialDataFrameMatrix = np.zeros((permutedArray.shape[0],len(COLUMNS_SWEEP)))
        initialDataFrameMatrix[:permutedArray.shape[0],:permutedArray.shape[1]] = permutedArray
        self.data = pd.DataFrame(initialDataFrameMatrix,columns=COLUMNS_SWEEP)

    def initializeParameters(self,description):
        self.parameter = dict.fromkeys(PARAMS)
        self.parameter["Measurement Date"] = datetime.datetime.now()
        self.parameter["Measurement Description"] = description
        self.parameter["Sampling Rate"] = 0
        self.parameter["Sampling Harmonic"] = 1
        self.parameter["Carrier Frequency"] = 985000
        self.parameter["Injection Frequency"] = 1000
        self.parameter["Channel Gain"] = {'X':1000,'Y':1000}

    def updateInstance(self,coords,data):
        index = self.data[
            (self.data['Pressure']==coords['Pressure']) &
            (self.data['Dipole']==coords['Dipole'])].index.tolist()[0]
        print(f'Measurement index: {index}')
        amp_X, ph_X = self._parseFFT(data['Demod X'],self.parameter["Injection Frequency"],self.parameter["Channel Gain"]['X'],self.parameter["Sampling Rate"])
        amp_Y, ph_Y = self._parseFFT(data['Demod Y'],self.parameter["Injection Frequency"],self.parameter["Channel Gain"]['Y'],self.parameter["Sampling Rate"])
        # amp_X, ph_X, amp_Y, ph_Y = 1,1,1,1
        self.data.at[index,'Amplitude X'] = amp_X
        self.data.at[index,'Amplitude Y'] = amp_Y
        self.data.at[index,'Phase X'] = ph_X
        self.data.at[index,'Phase Y'] = ph_Y
        print(f'Measurement instance: {amp_X}:{amp_Y}:{ph_X}:{ph_Y}')

    def _find_nearest_bin(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def _parseFFT(self,data,frequency,gain,smplrate):
        np_fft_val = np.fft.fft(data)
        np_fft_freq = np.fft.fftfreq(len(data), d = 1/smplrate)
        np_fft_maxVal_idx = self._find_nearest_bin(np_fft_freq,frequency)
        
        amplitude = (2 / len(data) * np.abs(np_fft_val[np_fft_maxVal_idx])) 
        phase = np.angle(np_fft_val[np_fft_maxVal_idx])

        return amplitude/gain, phase

if __name__ == '__main__':
    pass