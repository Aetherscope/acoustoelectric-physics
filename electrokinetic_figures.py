#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Christopher Chare
# Created Date: 31 July 2025
# version ='1.0'
# -----

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import scienceplots
plt.style.use(['science','nature'])
plt.rcParams.update({
    "text.usetex": False,
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "font.size": 7,
    "mathtext.default": "regular",
    "axes.labelpad": 1
})

from modules.measurements import *
warnings.filterwarnings('ignore')

def fetchMatrix(primary,coords,component,measurement):
    df = measurement.data.loc[(measurement.data[coords[0]]==primary)]
    return  df.pivot(index=coords[1],columns=coords[2],values=f'Amplitude_{component}').values, \
            df.pivot(index=coords[1],columns=coords[2],values=f'Phase_{component}').values, \
            df.pivot(index=coords[1],columns=coords[2],values=f'Phasor_{component}').values, \
            df.pivot(index=coords[1],columns=coords[2],values=f'Amplitude_X_{component}').values, \
            df.pivot(index=coords[1],columns=coords[2],values=f'Amplitude_Y_{component}').values, \
            df.pivot(index=coords[1],columns=coords[2],values=f'Phase_X_{component}').values, \
            df.pivot(index=coords[1],columns=coords[2],values=f'Phase_Y_{component}').values

ANGLES = ['90','75','60','45','30','15','00']
FREQ = 0

def main():
    # === Load simulation and measurement data for interaction phantom, generate acoustoelectric potential field plots ===
    data_ampratio = []
    print("Generating electrokinetic figures...")
    for idx in range(len(ANGLES)):
        try:
            print(f"Parsing angle: {ANGLES[idx]}")
            with open(f"./measurements/measurement_{ANGLES[idx]}.ppk", "rb") as f:
                measurement = pickle.load(f)

            df_sim_ae_field = pd.read_csv(f"./simulations/simulation_{ANGLES[idx]}.txt",sep='\s+',skiprows=9,names=['X','Y','Vae'],na_values=['NaN'])
            df_sim_ae_field['Vae'] = df_sim_ae_field['Vae'].str.replace('i','j').apply(lambda x: np.complex128(x))
            df_sim_ae_field[['X','Y']] = df_sim_ae_field[['X','Y']].round(2)
        except:
            print('Unable to open data...')
        else:
            primaryAxis, ternaryAxis, matCoords = 'X', 'Z', ['Y','X','Z']
            x, y = np.unique(measurement.data[primaryAxis].values), np.unique(measurement.data[ternaryAxis].values)
            X, Y = np.meshgrid(x, y,indexing='ij')
            amp, phs, phr, _, _, _, _= fetchMatrix(0,matCoords,f'{FREQ}',measurement)

            fig, ax = plt.subplots(1,1,figsize=(2.5,1.6))
            c = ax.imshow(amp.T*1e3, cmap='bone',extent=[np.min(x),np.max(x),np.min(y),np.max(y)],interpolation='nearest',origin='lower')
            ax.set_xlabel('$\it x$ (mm)')
            ax.set_ylabel('$\it z$ (mm)')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) 
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            ax.tick_params(colors='white', which='both', labelcolor='black')
            ax.set_aspect('equal', 'box')
            cbar = fig.colorbar(c, ax=ax)
            cbar.set_label(f'Amplitude (mV)',labelpad=3)
            fig.tight_layout()
            filename = f"./figures/electrokinetic_{ANGLES[idx]}.pdf"
            fig.savefig(filename,dpi=300)
            print(f"Figure saved as {filename}")

if __name__ == "__main__":
    main()