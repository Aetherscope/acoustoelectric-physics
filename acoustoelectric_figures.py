#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Christopher Chare
# Created Date: 31 July 2025
# version ='1.0'
# -----

from os import listdir
from os.path import isdir, join
import numpy as np
import scipy as sp
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import cmocean
from scipy.optimize import minimize
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

def fetchField(primary,coords,measurement):
    df = measurement.data.loc[(measurement.data[coords[0]]==primary)]
    return  df.pivot(index=coords[1],columns=coords[2],values=f'Amplitude X').values, \
            df.pivot(index=coords[1],columns=coords[2],values=f'Amplitude Y').values, \
            df.pivot(index=coords[1],columns=coords[2],values=f'Phase X').values, \
            df.pivot(index=coords[1],columns=coords[2],values=f'Phase Y').values

def extractMatrix(gridpts,df_sim_ae_field):
    df = df_sim_ae_field[(df_sim_ae_field.X.isin(gridpts[0])) &
                         (df_sim_ae_field.Y.isin(gridpts[1]))]
    return  df.pivot(index='X',columns='Y',values='Vae').values

def extractField(primary,coords,parameter,dataframe):
    df = dataframe.loc[(dataframe[coords[0]]==primary)]
    return  df.pivot(index=coords[1],columns=coords[2],values=parameter).values

def shiftMatrix(shift,x,y,df_sim_ae_field):
    return extractMatrix([np.round(x+shift[0]*0.01,2),np.round(y+shift[1]*0.01,2)],df_sim_ae_field)

def costFunction(shift,amp,x,y,df_sim_ae_field):
    shiftedAmp = np.abs(shiftMatrix(shift,x,y,df_sim_ae_field))
    return np.nansum((amp-shiftedAmp*shift[2])**2)

def gaussian_blur(U,sigma,truncate):
    V=U.copy()
    V[np.isnan(U)]=0
    VV=sp.ndimage.gaussian_filter(V,sigma=sigma,truncate=truncate)
    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW=sp.ndimage.gaussian_filter(W,sigma=sigma,truncate=truncate)
    return VV/WW

def _parseConversionFactor(frequency=985000):
    try:
        filePath = './calibrations'
        fileName = "PrecisionAcoustics PA18103 SN3495.txt"
        calData = np.loadtxt(f"{filePath}/{fileName}",skiprows=10,delimiter=',',dtype=float)
        calFactor = np.interp(frequency/1e6,calData[:,0],calData[:,1]) / 1000 # Convert mV/MPa into (--)mV/kPa
    except Exception as e:
        print(f"Unable to load calibration data :: {e}")
        return False
    else:
        return calFactor

ANGLES = ['90','75','60','45','30','15','00','LF']
FREQ = 1000

def main():
    # === Load simulation and measurement data for interaction phantom, generate acoustoelectric potential field plots ===
    data_ampratio = []
    print("Generating acoustoelectric interaction phantom figures...")
    for idx in range(len(ANGLES)-1):
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

            res = minimize(costFunction,x0=[0,0,1],args=(amp,x,y,df_sim_ae_field),method='Powell',bounds=((-50, 50), (-50, 50),(0,10)))
            print(f"Alignment x: {np.round(res.x[0])} px | y: {np.round(res.x[1])} px")
            simAmp = np.abs(shiftMatrix(res.x,x,y,df_sim_ae_field))
            simAmpPhs = np.angle(shiftMatrix(res.x,x,y,df_sim_ae_field))

            data_ampratio.append((amp,simAmp))

            fig, ax = plt.subplots(2,1,figsize=(2.5,2.7))
            c = ax[0].imshow((amp*1e6 * np.cos(phs)).T , cmap=cmocean.cm.balance, vmin=-np.nanmax(amp*1e6), vmax=np.nanmax(amp*1e6),extent=[np.min(x),np.max(x),np.min(y),np.max(y)],interpolation='nearest',origin='lower')
            ax[0].set_xlabel('$\it x$ (mm)')
            ax[0].set_ylabel('$\it z$ (mm)')
            ax[0].xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax[0].yaxis.set_major_locator(ticker.MultipleLocator(5)) 
            ax[0].xaxis.label.set_color('black')
            ax[0].yaxis.label.set_color('black')
            ax[0].tick_params(colors='black', which='both', labelcolor='black')
            ax[0].set_aspect('equal', 'box')
            cbar = fig.colorbar(c, ax=ax[0],format="%i")
            cbar.set_label('$\it V_{\mathrm{ae,exp}}$ ($\mu$V)',labelpad=3)

            c = ax[1].imshow((simAmp*1e6 * np.cos(simAmpPhs)).T , cmap=cmocean.cm.balance, vmin=-np.nanmax(simAmp*1e6), vmax=np.nanmax(simAmp*1e6),extent=[np.min(x),np.max(x),np.min(y),np.max(y)],interpolation='nearest',origin='lower')
            ax[1].set_xlabel('$\it x$ (mm)')
            ax[1].set_ylabel('$\it z$ (mm)')
            ax[1].xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax[1].yaxis.set_major_locator(ticker.MultipleLocator(5)) 
            ax[1].xaxis.label.set_color('black')
            ax[1].yaxis.label.set_color('black')
            ax[1].tick_params(colors='black', which='both', labelcolor='black')
            ax[1].set_aspect('equal', 'box')
            cbar = fig.colorbar(c, ax=ax[1],format="%i")
            cbar.set_label('$\it V_{\mathrm{ae,sim}}$ ($\mu$V)',labelpad=3)
            fig.tight_layout()
            filename = f"./figures/interaction_phantom_{ANGLES[idx]}.pdf"
            fig.savefig(filename,dpi=300)
            print(f"Figure saved as {filename}")

    # === Load simulation and measurement data for leadfield phantom, generate acoustoelectric potential field plots ===
    print("Generating acoustoelectric leadfield phantom figures...")
    with open("./measurements/measurement_LF.ppk", "rb") as f:
        measurement = pickle.load(f)
    primaryAxis, ternaryAxis, matCoords = 'X', 'Y', ['Z','Y','X']
    x_m, y_m = np.unique(measurement.data[primaryAxis].values), np.unique(measurement.data[ternaryAxis].values)
    X, Y = np.meshgrid(x_m, y_m,indexing='ij')
    amp, phs, phr, _, _, _, _= fetchMatrix(0,matCoords,f'{FREQ}',measurement)

    fig, ax = plt.subplots(1,1,figsize=(2.5,1.6))
    c = ax.imshow(amp.T*1e6, cmap=cmocean.cm.deep_r,vmin=0,extent=[np.min(x_m),np.max(x_m),np.min(y_m),np.max(y_m)],interpolation='nearest',origin='lower')
    ax.set_xlabel('$\it x$ (mm)')
    ax.set_ylabel('$\it y$ (mm)')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) 
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(colors='white', which='both', labelcolor='black')
    ax.set_aspect('equal', 'box')
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('$|\it V_{\mathrm{ae,LF}}|$ ($\mu$V)',labelpad=3)
    ax.set_title('Experiment',fontdict={'family':'sans-serif','size':7})
    fig.tight_layout()
    filename = './figures/leadfield_phantom_measurement.pdf'
    fig.savefig(filename,dpi=300)
    print(f"Figure saved as {filename}")

    condition = ["floating","dirichlet"]
    fonttitle = {'family':'sans-serif','size':7}
    for idx in range(len(condition)):
        df_sweep_lf = pd.read_csv(f"./simulations/simulation_LF_{condition[idx]}.txt",sep='\s+',skiprows=5,names=['X','Z','LF','LFCORR'],na_values=['NaN'])
        df_sweep_lf['LF'] = df_sweep_lf['LF'].str.replace('i','j').apply(lambda x: np.complex128(x))
        df_sweep_lf['LFCORR'] = df_sweep_lf['LFCORR'].str.replace('i','j').apply(lambda x: np.complex128(x))
        df_sweep_lf[['X','Z']] = df_sweep_lf[['X','Z']].round(2)
        sweepx,sweepz = np.unique(df_sweep_lf['X']),np.unique(df_sweep_lf['Z'])
        n_sweepx,n_sweepz = len(sweepx),len(sweepz)

        sweepLF = np.reshape(df_sweep_lf['LF'].values,(n_sweepx,n_sweepz))
        fig, ax = plt.subplots(1,1,figsize=(2.5,1.6))
        c = ax.imshow(np.abs(-sweepLF).T*1e6, cmap=cmocean.cm.deep_r,extent=[np.min(sweepx),np.max(sweepx),np.min(sweepz),np.max(sweepz)],interpolation='nearest',origin='lower')
        ax.set_xlabel('$\it x$ (mm)')
        ax.set_ylabel('$\it y$ (mm)')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(colors='white', which='both', labelcolor='black')
        ax.set_aspect('equal', 'box')
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('$|\it V_{\mathrm{ae,LF}}|$ ($\mu$V)',labelpad=3)
        ax.set_title('$\it V_\\mathrm{ae,LF}=-\int \it \\rho \it k_\\mathrm{ae} \it P (\mathbf{J}_0 \cdot \mathbf{J}_\mathrm{L}) \,\mathrm{d}v$',fontdict=fonttitle)
        fig.tight_layout()
        filename = f'./figures/leadfield_phantom_simulation_original_{condition[idx]}.pdf'
        fig.savefig(filename,dpi=300)
        print(f"Figure saved as {filename}")

        sweepLFCORR = np.reshape(df_sweep_lf['LFCORR'].values,(n_sweepx,n_sweepz))
        fig, ax = plt.subplots(1,1,figsize=(2.5,1.6))
        c = ax.imshow(np.abs(-sweepLFCORR).T*1e6, cmap=cmocean.cm.deep_r,extent=[np.min(sweepx),np.max(sweepx),np.min(sweepz),np.max(sweepz)],interpolation='nearest',origin='lower')
        ax.set_xlabel('$\it x$ (mm)')
        ax.set_ylabel('$\it y$ (mm)')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(colors='white', which='both', labelcolor='black')
        ax.set_aspect('equal', 'box')
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('$|\it V_{\mathrm{ae,LF}}|$ ($\mu$V)',labelpad=3)
        ax.set_title('$\it V_\\mathrm{ae,LF}=-\int \it k_\\mathrm{ae} (\\nabla \it P \cdot \mathbf{J}_0) Z_\mathrm{L} \,\mathrm{d}v$',fontdict=fonttitle)
        fig.tight_layout()
        filename = f'./figures/leadfield_phantom_simulation_revised_{condition[idx]}.pdf'
        fig.savefig(filename,dpi=300)
        print(f"Figure saved as {filename}")

        df_sweep_pois = pd.read_csv(f"./simulations/simulation_POISSON_{condition[idx]}.txt",sep='\s+',skiprows=5,names=['Z','X','POISSON'],na_values=['NaN'])
        df_sweep_pois['POISSON'] = df_sweep_pois['POISSON'].str.replace('i','j').apply(lambda x: np.complex128(x))
        df_sweep_pois[['X','Z']] = df_sweep_pois[['X','Z']].round(2)
        poisx,poisz = np.unique(df_sweep_pois['X']),np.unique(df_sweep_pois['Z'])
        n_poisx,n_poisz = len(poisx),len(poisz)
        sweepPOIS = np.reshape(df_sweep_pois['POISSON'].values,(n_poisx,n_poisz))

        fig, ax = plt.subplots(1,1,figsize=(2.5,1.6))
        c = ax.imshow(np.abs(-sweepPOIS).T*1e6, cmap=cmocean.cm.deep_r,extent=[np.min(sweepx),np.max(sweepx),np.min(sweepz),np.max(sweepz)],interpolation='nearest',origin='lower')
        ax.set_xlabel('$\it x$ (mm)')
        ax.set_ylabel('$\it y$ (mm)')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(colors='white', which='both', labelcolor='black')
        ax.set_aspect('equal', 'box')
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('$|\it V_{\mathrm{ae,LF}}|$ ($\mu$V)',labelpad=3)
        ax.set_title('$\\nabla^2 \it V_\\mathrm{ae}= \it k_\\mathrm{ae} (\\nabla \it P \cdot \mathbf{E}_0)$',fontdict=fonttitle)
        fig.tight_layout()
        filename = f'./figures/leadfield_phantom_simulation_poisson_{condition[idx]}.pdf'
        fig.savefig(filename,dpi=300)
        print(f"Figure saved as {filename}")

    interp = RegularGridInterpolator((poisx,poisz), np.abs(sweepPOIS))
    simAmp = np.reshape(interp(np.column_stack((np.ravel(X),np.ravel(Y)))),(len(x_m),len(y_m))).T
    data_ampratio.append((amp,simAmp))

    # === Load control measurement without pressure field to extract injected background noise from the dipole source ===
    print("Generating acoustoelectric interaction phantom background noise figure...")
    with open("./measurements/measurement_90_nopressure.ppk", "rb") as f:
        meas_nopres = pickle.load(f)
    primaryAxis, ternaryAxis, matCoords = 'X', 'Z', ['Y','X','Z']
    x, y = np.unique(meas_nopres.data[primaryAxis].values), np.unique(meas_nopres.data[ternaryAxis].values)
    X, Y = np.meshgrid(x, y,indexing='ij')
    amp_nopres, phs, phr, _, _, _, _= fetchMatrix(0,matCoords,1000,meas_nopres)
    sigma = 6
    blurred_bg = gaussian_blur(amp_nopres,sigma=sigma,truncate=sigma)
    blurred_bg[np.isnan(amp_nopres)]=np.nan
    fig, ax = plt.subplots(1,1,figsize=(3,1.5))
    c = ax.imshow(blurred_bg.T*1e6, cmap=cmocean.cm.deep_r, vmin=0,vmax=0.5,extent=[np.min(x),np.max(x),np.min(y),np.max(y)],interpolation='nearest',origin='lower')
    ax.set_xlabel('$\it x$ (mm)')
    ax.set_ylabel('$\it z$ (mm)')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) 
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(colors='white', which='both', labelcolor='black')
    ax.set_aspect('equal', 'box')
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label(f'Amplitude ($\mu$V)',labelpad=3)
    fig.tight_layout()
    filename = './figures/interaction_phantom_nopressure.pdf'
    fig.savefig(filename,dpi=300)
    print(f"Figure saved as {filename}")

    # === Parse acoustoelectric potential data, extract acoustoelectric interaction coefficient ===
    print("Generating acoustoelectric interaction coefficient figures...")
    fig, ax = plt.subplots(1,1,figsize=(3,3))
    for idx in range(len(ANGLES)-1):
        ax.scatter(data_ampratio[idx][1],data_ampratio[idx][0],marker='.',s=1,label=f"{ANGLES[idx]}$^\circ$",alpha=1)
    ax.scatter(data_ampratio[-1][1],data_ampratio[-1][0],marker='.',s=1,label=f"{ANGLES[-1]}",alpha=1)
    ax.set_xlim(1e-9,1e-4)
    ax.set_ylim(1e-9,1e-4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$|\it V_{\mathrm{ae,sim}}|$ (V)')
    ax.set_ylabel('$|\it V_{\mathrm{ae,exp}}|$ (V)')
    ax.legend(
        ncol=4, 
        loc='upper center', 
        bbox_to_anchor=(0.4, 1.3),  
        frameon=False
    )
    fig.tight_layout()
    filename = './figures/scatter_acoustoelectric_potential.pdf'
    fig.savefig(filename,dpi=300)
    print(f"Figure saved as {filename}")

    NOISE = 4.5271196887966336e-08
    fig, ax = plt.subplots(1,1,figsize=(3,3))
    for idx in range(len(ANGLES)-1):
        tempratio = data_ampratio[idx][0]
        tempratio[tempratio<=blurred_bg] = np.nan
        tempratio = data_ampratio[idx][1]
        tempratio[tempratio<=blurred_bg] = np.nan
        ax.scatter(data_ampratio[idx][1],data_ampratio[idx][0],marker='.',s=1,label=f"{ANGLES[idx]}$^\circ$",alpha=1)
    tempratio = data_ampratio[-1][0]
    tempratio[tempratio<=NOISE] = np.nan
    tempratio = data_ampratio[-1][1]
    tempratio[tempratio<=NOISE] = np.nan
    ax.scatter(data_ampratio[-1][1],data_ampratio[-1][0],marker='.',s=1,label=f"{ANGLES[-1]}",alpha=1)
    ax.set_xlim(1e-9,1e-4)
    ax.set_ylim(1e-9,1e-4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$|\it V_{\mathrm{ae,sim}}|$ (V)')
    ax.set_ylabel('$|\it V_{\mathrm{ae,exp}}|$ (V)')
    ax.legend(
        ncol=4, 
        loc='upper center', 
        bbox_to_anchor=(0.4, 1.3),  
        frameon=False
    )
    fig.tight_layout()
    filename = './figures/scatter_acoustoelectric_potential_masked.pdf'
    fig.savefig(filename,dpi=300)
    print(f"Figure saved as {filename}")

    plt.figure(figsize=(3,3))
    for idx in range(len(ANGLES)):
        mulAmp = np.copy(data_ampratio[idx][0])
        mulsimAmp = np.copy(data_ampratio[idx][1])
        mulArray = (mulAmp/mulsimAmp).flatten()
        mulArrayflat = np.hstack(mulArray);
        hist, bins = np.histogram(mulArrayflat[~np.isnan(mulArrayflat)],bins=np.logspace(np.log10(0.1),np.log10(10.0),50))
        if ANGLES[idx] == "LF":
            plt.stairs(hist, bins,hatch='',alpha=1,label=f"{ANGLES[idx]}",fill=False,ls='--')
        else:
            plt.stairs(hist, bins,hatch='',alpha=1,label=f"{ANGLES[idx]}$^\circ$")
    plt.xscale('log')
    plt.ylabel('$|\it V_{\mathrm{ae,exp}}/\it V_{\mathrm{ae,sim}}|$')
    plt.xlabel('$\it k_\mathrm{ae}$ ($\\times 10^{-9}~\mathrm{Pa}^{-1}$)')
    plt.legend()
    filename = './figures/histogram_acoustoelectric_coefficient.pdf'
    plt.savefig(filename,dpi=300)
    print(f"Figure saved as {filename}")

    data_ampratio_cropped = []
    data_ampratio_geomeans = []
    data_ampratio_geostds = []
    for idx in range(len(ANGLES)):
        mulAmp = np.copy(data_ampratio[idx][0])
        mulsimAmp = np.copy(data_ampratio[idx][1])
        mulArray = (mulAmp/mulsimAmp).flatten()
        data_ampratio_cropped.append(mulArray[~np.isnan(mulArray)])
        data_ampratio_geomeans.append(np.mean(np.log(mulArray[~np.isnan(mulArray)])))
        data_ampratio_geostds.append(np.std(np.log(mulArray[~np.isnan(mulArray)])))
    data_ampratio_logmean = np.mean(np.log(np.concatenate(data_ampratio_cropped).ravel()))
    data_ampratio_logstd = np.std(np.log(np.concatenate(data_ampratio_cropped).ravel()))

    cmap_div = ['#F4E7C5FF', '#023743FF', '#72874EFF', '#476F84FF', '#ACC2CFFF', '#453947FF','#FED789FF']
    fig, ax = plt.subplots(1,2,figsize=(3.6,2),sharey=True,gridspec_kw={'width_ratios': [7, 1]})
    ax[0].axhline(np.exp(data_ampratio_logmean),linestyle='-',label='This work (mean)',linewidth=1,color=cmap_div[1],alpha=1)
    ax[0].fill_between([-1,8], [np.exp(data_ampratio_logmean-data_ampratio_logstd),np.exp(data_ampratio_logmean-data_ampratio_logstd)],
                    [np.exp(data_ampratio_logmean+data_ampratio_logstd),np.exp(data_ampratio_logmean+data_ampratio_logstd)],
                    color=cmap_div[4],linewidth=0,edgecolor=cmap_div[0],alpha=1,label='This work (stddev)')
    ax[0].axhline(0.73,linestyle=(0, (3, 1, 1, 1)),label='K$\ddot{o}$rber (1909)',color=cmap_div[3])
    ax[0].axhline(1.086,linestyle='-.',label='Fox et al. (1946)',color=cmap_div[3])
    ax[0].axhline(0.967,linestyle='--',label='Jossinet et al. (1998)',color=cmap_div[3])
    ax[0].fill_between([-1,8], [1*1/0.77,1*1/0.77],
                    [1*1/0.5,1*1/0.5],
                    color=cmap_div[0],linewidth=0,hatch='/////',edgecolor=cmap_div[6],alpha=1,label='Lavandier et al. (2000)')
    ax[0].axhline(0.34,linestyle=(0, (1, 1)),label='Li et al. (2012)',color=cmap_div[3],linewidth=1.5)
    ax[0].errorbar(np.flip(np.arange(len(ANGLES)-1))+1,np.exp(data_ampratio_geomeans[0:len(ANGLES)-1]),
                [np.abs(np.exp(np.array(data_ampratio_geomeans[0:len(ANGLES)-1])-np.array(data_ampratio_geostds[0:len(ANGLES)-1]))-np.exp(data_ampratio_geomeans[0:len(ANGLES)-1])),
                np.exp(np.array(data_ampratio_geomeans[0:len(ANGLES)-1])+np.array(data_ampratio_geostds[0:len(ANGLES)-1]))-np.exp(data_ampratio_geomeans[0:len(ANGLES)-1])],
                fmt='s',markersize=3,capsize=5,capthick=1,color=cmap_div[1],elinewidth=1)
    ax[0].set_xticks([7,6,5,4,3,2,1])
    ax[0].set_xticklabels(['90$^\circ$','75$^\circ$','60$^\circ$','45$^\circ$','30$^\circ$','15$^\circ$','0$^\circ$'])
    ax[0].set_xlim(0.5,7.5)
    ax[0].set_ylim(0,2.2)
    ax[0].set_xlabel('Ultrasound incidence angle $\it\\theta$')
    ax[0].set_ylabel('$\it k_{ae}$ ($\\times 10^{-9}~\mathrm{Pa}^{-1}$)')
    ax[1].axhline(np.exp(data_ampratio_logmean),linestyle='-',linewidth=1,color=cmap_div[1],alpha=1)
    ax[1].fill_between([-1,8], [np.exp(data_ampratio_logmean-data_ampratio_logstd),np.exp(data_ampratio_logmean-data_ampratio_logstd)],
                    [np.exp(data_ampratio_logmean+data_ampratio_logstd),np.exp(data_ampratio_logmean+data_ampratio_logstd)],
                    color=cmap_div[4],linewidth=0,edgecolor=cmap_div[0],alpha=1)
    ax[1].axhline(0.73,linestyle='-.',color=cmap_div[3])
    ax[1].axhline(1.086,linestyle='-.',color=cmap_div[3])
    ax[1].axhline(0.967,linestyle='--',color=cmap_div[3])
    ax[1].fill_between([-1,8], [1*1/0.77,1*1/0.77],
                    [1*1/0.5,1*1/0.5],
                    color=cmap_div[0],linewidth=0,hatch='/////',edgecolor=cmap_div[6],alpha=1)
    ax[1].axhline(0.34,linestyle=(0, (1, 1)),color=cmap_div[3],linewidth=1.5)

    ax[1].errorbar([0],np.exp(data_ampratio_geomeans[-1]),yerr=
                    np.array([np.abs(np.exp(np.array(data_ampratio_geomeans[-1])-np.array(data_ampratio_geostds[-1]))-np.exp(data_ampratio_geomeans[-1])),
                    np.exp(np.array(data_ampratio_geomeans[-1])+np.array(data_ampratio_geostds[-1]))-np.exp(data_ampratio_geomeans[-1])])[:,None],
                    fmt='s',markersize=3,capsize=5,capthick=1,color=cmap_div[1],elinewidth=1)
    ax[1].set_xlim(-0.5,0.5)
    ax[1].set_xticks([0])
    ax[1].set_xticklabels(['LF'])
    fig.legend(loc='upper center', bbox_to_anchor=(0.49, 1.3), ncol=2)
    fig.tight_layout()
    filename = './figures/acoustoelectric_interaction_coefficients.pdf'
    fig.savefig(filename,dpi=300)
    print(f"Figure saved as {filename}")

    # === Load simulation and measurement data for interaction phantom, generate pressure and potential field plots ===
    print("Generating acoustoelectric interaction phantom measured pressure and electric fields...")
    df_sim_p_field = pd.read_csv('./simulations/simulation_pressure_field.txt',sep='\s+',skiprows=9,names=['X','Y','Z','P'],na_values=['NaN']) 
    df_sim_p_field['P'] = df_sim_p_field['P'].str.replace('i','j').apply(lambda x: np.complex128(x))
    df_sim_p_field[['X','Y','Z']] = df_sim_p_field[['X','Y','Z']].round(2)
    df_sim_p_field.drop_duplicates(subset=['X','Y','Z'],keep='first',inplace=True)
    primaryAxis, ternaryAxis, matCoords = 'X', 'Z', ['Y','X','Z']
    simx, simy = np.unique(df_sim_p_field[primaryAxis].values), np.unique(df_sim_p_field[ternaryAxis].values)
    simAmp = extractField(0,matCoords,'P',df_sim_p_field)
    simAmplitude = np.abs(simAmp) # Not transposed when using XZ
    simPhase = np.angle(simAmp)

    fig, ax = plt.subplots(1,1,figsize=(2.5,1.5))
    c = ax.imshow((simAmplitude*np.cos(simPhase)) * 1e-6, cmap=cmocean.cm.balance,extent=[np.min(simy),np.max(simy),np.min(simx),np.max(simx)],interpolation='nearest',origin='upper')
    axislimit = 7.5
    ax.axis([-axislimit, axislimit, -axislimit, axislimit])
    ax.set_xlabel('$\it y$ (mm)')
    ax.set_ylabel('$\it z$ (mm)')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) 
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(colors='black', which='both', labelcolor='black')
    ax.set_aspect('equal', 'box')
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Pressure (MPa)',labelpad=3)
    fig.tight_layout()
    filename = './figures/interaction_phantom_pressure_simulated.pdf'
    fig.savefig(filename,dpi=300)
    print(f"Figure saved as {filename}")

    with open("./measurements/measurement_pressure_field.pkl", "rb") as f:
        measurement = pickle.load(f)
    primaryAxis, ternaryAxis, matCoords = 'X', 'Z', ['Y','X','Z']
    x, y = np.unique(measurement.data[primaryAxis].values), np.unique(measurement.data[ternaryAxis].values)
    ampX, ampY, phX, phY = fetchField(0,matCoords,measurement)
    sensitivity = _parseConversionFactor(frequency=985000)
    measAmplitude = 0.5*ampX / (sensitivity/1e6)
    fig, ax = plt.subplots(1,1,figsize=(2.5,1.5))
    c = ax.imshow((measAmplitude*np.cos(phX)).T * 1e-6, cmap=cmocean.cm.balance,extent=[np.min(x),np.max(x),np.min(y)+0.5,np.max(y)+0.5],interpolation='nearest',origin='upper')
    axislimit = 7.5
    ax.axis([-axislimit, axislimit, -axislimit, axislimit])
    ax.set_xlabel('$\it y$ (mm)')
    ax.set_ylabel('$\it z$ (mm)')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(colors='black', which='both', labelcolor='black')
    ax.set_aspect('equal', 'box')
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Pressure (MPa)',labelpad=3)
    fig.tight_layout()
    filename = './figures/interaction_phantom_pressure_measured.pdf'
    fig.savefig(filename,dpi=300)
    print(f"Figure saved as {filename}")

    with open("./measurements/measurement_potential_field.pkl", "rb") as f:
        measurement = pickle.load(f)
    step_size = 0.15e-3
    primaryAxis, ternaryAxis, matCoords = 'X', 'Z', ['Y','X','Z']
    x, y = np.unique(measurement.data[primaryAxis].values), np.unique(measurement.data[ternaryAxis].values)
    ampX, ampY, phX, phY = fetchField(0,matCoords,measurement)
    phasor = ampX*np.exp(1j*phX) - ampY*np.exp(1j*phY)
    amplitude = np.abs(phasor)
    phase = np.angle(phasor)
    [gx,gy] = np.gradient(amplitude*np.exp(1j*phase))
    measAmplitude = np.sqrt(np.abs(gx)**2 + np.abs(gy)**2)/step_size
    fig, ax = plt.subplots(1,1,figsize=(2.5,1.5))
    c = ax.imshow(1e-3*measAmplitude.T, cmap=cmocean.cm.deep_r,alpha=1,vmin=0,vmax=0.3,extent=[np.min(x),np.max(x),np.min(y),np.max(y)],interpolation='nearest',origin='lower')
    ax.streamplot(x, y, 1e-3*np.real(gx).T, 1e-3*np.real(gy).T, color='w', linewidth=0.3, arrowsize=0.4, density=0.6)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    ax.set_xlabel('$\it x$ (mm)')
    ax.set_ylabel('$\it z$ (mm)')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) 
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(colors='white', which='both', labelcolor='black')
    ax.set_aspect('equal', 'box')
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Electric field (kV/m)',labelpad=3)
    fig.tight_layout()
    filename = './figures/interaction_phantom_electric_measured.pdf'
    fig.savefig(filename,dpi=300)
    print(f"Figure saved as {filename}")

    df_sim_e_field = pd.read_csv('./simulations/simulation_electric_field.txt',sep='\s+',skiprows=9,names=['X','Y','Z','Ex','Ey'],na_values=['NaN'])
    df_sim_e_field['E'] = np.real(np.sqrt(df_sim_e_field['Ex']**2 + df_sim_e_field['Ey']**2))
    df_sim_e_field.drop_duplicates(subset=['X','Y','Z'],keep='first',inplace=True)
    primaryAxis, ternaryAxis, matCoords = 'X', 'Y', ['Z','Y','X']
    simx, simy = np.unique(df_sim_e_field[primaryAxis].values), np.unique(df_sim_e_field[ternaryAxis].values)
    simAmplitude = extractField(0,matCoords,'E',df_sim_e_field).T
    simEx = -extractField(0,matCoords,'Ey',df_sim_e_field).T
    simEy = -extractField(0,matCoords,'Ex',df_sim_e_field).T
    fig, ax = plt.subplots(1,1,figsize=(2.5,1.5))
    c = ax.imshow(1e-3*simAmplitude, cmap=cmocean.cm.deep_r,alpha=1,vmin=0,vmax=0.3,extent=[np.min(simx),np.max(simx),np.min(simy),np.max(simy)],interpolation='nearest',origin='lower')
    ax.streamplot(simx, simy, -1e-3*simEx, -1e-3*simEy, color='w', linewidth=0.3, arrowsize=0.4, density=0.75)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    ax.set_xlabel('$\it x$ (mm)')
    ax.set_ylabel('$\it z$ (mm)')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) 
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(colors='white', which='both', labelcolor='black')
    ax.set_aspect('equal', 'box')
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Electric field (kV/m)',labelpad=3)
    fig.tight_layout()
    filename = './figures/interaction_phantom_electric_simulated.pdf'
    fig.savefig(filename,dpi=300)
    print(f"Figure saved as {filename}")


if __name__ == "__main__":
    main()