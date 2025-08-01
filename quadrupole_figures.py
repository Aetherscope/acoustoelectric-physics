#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Christopher Chare
# Created Date: 31 July 2025
# version ='1.0'
# -----

import numpy as np
import pandas as pd
import cmocean
import scienceplots
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
plt.style.use(['science','nature'])
plt.rcParams.update({
    "text.usetex": False,
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "font.size": 7,
    "mathtext.default": "regular",
    "axes.labelpad": 1
})
fonttitle = {'family':'sans-serif','size':7}

BOUNDARIES = ["floating","dirichlet"]

def main():
    
    for condition in BOUNDARIES:
        df_quad_lf = pd.read_csv(f'./simulations/quadrupole_LF_{condition}.txt',sep='\s+',skiprows=5,names=['X','Y','F','LF','LFCORR'],na_values=['NaN'])
        df_quad_lf['LF'] = df_quad_lf['LF'].str.replace('i','j').apply(lambda x: np.complex128(x))
        df_quad_lf['LFCORR'] = df_quad_lf['LFCORR'].str.replace('i','j').apply(lambda x: np.complex128(x))
        df_quad_lf[['X','Y']] = df_quad_lf[['X','Y']].round(2)
        df_quad_pois = pd.read_csv(f'./simulations/quadrupole_POISSON_{condition}.txt',sep='\s+',skiprows=9,names=['X','Y','POIS'],na_values=['NaN'])
        df_quad_pois['POIS'] = df_quad_pois['POIS'].str.replace('i','j').apply(lambda x: np.complex128(x))
        df_quad_pois[['X','Y']] = df_quad_pois[['X','Y']].round(2)

        x_lf,y_lf = df_quad_lf['X'].unique(),df_quad_lf['Y'].unique()
        xn_lf,yn_lf = len(x_lf),len(y_lf)
        simLF = np.reshape(df_quad_lf['LF'].values,(xn_lf,yn_lf))
        simLFCORR = np.reshape(df_quad_lf['LFCORR'].values,(xn_lf,yn_lf))
        x_pois,y_pois = df_quad_pois['X'].unique(),df_quad_pois['Y'].unique()
        xn_pois,yn_pois = len(x_pois),len(y_pois)
        simPOIS = np.reshape(df_quad_pois['POIS'].values,(xn_pois,yn_pois))

        # Mask the source nodes for clarity
        masked_simLF = simLF.copy()
        masked_simLF[30,30],masked_simLF[30,50],masked_simLF[50,30],masked_simLF[50,50] = np.nan,np.nan,np.nan,np.nan
        masked_simLFCORR = simLFCORR.copy()
        masked_simLFCORR[30,30],masked_simLFCORR[30,50],masked_simLFCORR[50,30],masked_simLFCORR[50,50] = np.nan,np.nan,np.nan,np.nan
        masked_simPOIS = simPOIS.copy()
        masked_simPOIS[36,60],masked_simPOIS[36,36],masked_simPOIS[60,36],masked_simPOIS[60,60] = np.nan,np.nan,np.nan,np.nan

        fig, ax = plt.subplots(1,1,figsize=(2.5,1.6))
        c = ax.imshow(np.real(-masked_simLF).T*1e6, cmap=cmocean.cm.balance,extent=[np.min(x_lf),np.max(x_lf),np.min(y_lf),np.max(y_lf)],interpolation='nearest',origin='lower')
        ax.set_xlabel('$\it x$ (mm)')
        ax.set_ylabel('$\it y$ (mm)')
        ax.set_xlim(min(x_lf),max(x_lf))
        ax.set_ylim(min(y_lf),max(y_lf))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) 
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(colors='k', which='both', labelcolor='black')
        ax.set_aspect('equal', 'box')
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('$Re(\it V_{\mathrm{ae,LF}})$ ($\mu$V)',labelpad=3)
        ax.set_title('$\it V_\\mathrm{ae,LF}=-\int \it \\rho \it k_\\mathrm{ae} \it P (\mathbf{J}_0 \cdot \mathbf{J}_\mathrm{L}) \,\mathrm{d}v$',fontdict=fonttitle)
        fig.tight_layout()
        filename = f'./figures/quadrupole_simulation_original_{condition}_real.pdf'
        fig.savefig(filename,dpi=300)
        print(f"Figure saved as {filename}")

        fig, ax = plt.subplots(1,1,figsize=(2.5,1.6))
        c = ax.imshow(np.imag(-masked_simLF).T*1e6, cmap=cmocean.cm.balance,extent=[np.min(x_lf),np.max(x_lf),np.min(y_lf),np.max(y_lf)],interpolation='nearest',origin='lower')
        ax.set_xlabel('$\it x$ (mm)')
        ax.set_ylabel('$\it y$ (mm)')
        ax.set_xlim(min(x_lf),max(x_lf))
        ax.set_ylim(min(y_lf),max(y_lf))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) 
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(colors='k', which='both', labelcolor='black')
        ax.set_aspect('equal', 'box')
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('$Im(\it V_{\mathrm{ae,LF}})$ ($\mu$V)',labelpad=3)
        ax.set_title('$\it V_\\mathrm{ae,LF}=-\int \it \\rho \it k_\\mathrm{ae} \it P (\mathbf{J}_0 \cdot \mathbf{J}_\mathrm{L}) \,\mathrm{d}v$',fontdict=fonttitle)
        fig.tight_layout()
        filename = f'./figures/quadrupole_simulation_original_{condition}_imag.pdf'
        fig.savefig(filename,dpi=300)
        print(f"Figure saved as {filename}")

        fig, ax = plt.subplots(1,1,figsize=(2.5,1.6))
        c = ax.imshow(np.real(-masked_simLFCORR).T*1e6, cmap=cmocean.cm.balance,alpha=1,vmin=np.nanmin(np.real(-masked_simPOIS)*1e6),vmax=np.nanmax(np.real(-masked_simPOIS)*1e6),extent=[np.min(x_lf),np.max(x_lf),np.min(y_lf),np.max(y_lf)],interpolation='nearest',origin='lower')
        ax.set_xlabel('$\it x$ (mm)')
        ax.set_ylabel('$\it y$ (mm)')
        ax.set_xlim(min(x_lf),max(x_lf))
        ax.set_ylim(min(y_lf),max(y_lf))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) 
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(colors='k', which='both', labelcolor='black')
        ax.set_aspect('equal', 'box')
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('$Re(\it V_{\mathrm{ae,LF}})$ ($\mu$V)',labelpad=3)
        ax.set_title('$\it V_\\mathrm{ae,LF}=-\int \it k_\\mathrm{ae} (\\nabla \it P \cdot \mathbf{J}_0) Z_\mathrm{L} \,\mathrm{d}v$',fontdict=fonttitle)
        fig.tight_layout()
        filename = f'./figures/quadrupole_simulation_revised_{condition}_real.pdf'
        fig.savefig(filename,dpi=300)
        print(f"Figure saved as {filename}")

        fig, ax = plt.subplots(1,1,figsize=(2.5,1.6))
        c = ax.imshow(np.imag(-masked_simLFCORR).T*1e6, cmap=cmocean.cm.balance,alpha=1,vmin=np.nanmin(np.imag(-masked_simPOIS)*1e6),vmax=np.nanmax(np.imag(-masked_simPOIS)*1e6),extent=[np.min(x_lf),np.max(x_lf),np.min(y_lf),np.max(y_lf)],interpolation='nearest',origin='lower')
        ax.set_xlabel('$\it x$ (mm)')
        ax.set_ylabel('$\it y$ (mm)')
        ax.set_xlim(min(x_lf),max(x_lf))
        ax.set_ylim(min(y_lf),max(y_lf))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) 
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(colors='k', which='both', labelcolor='black')
        ax.set_aspect('equal', 'box')
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('$Im(\it V_{\mathrm{ae,LF}})$ ($\mu$V)',labelpad=3)
        ax.set_title('$\it V_\\mathrm{ae,LF}=-\int \it k_\\mathrm{ae} (\\nabla \it P \cdot \mathbf{J}_0) Z_\mathrm{L} \,\mathrm{d}v$',fontdict=fonttitle)
        fig.tight_layout()
        filename = f'./figures/quadrupole_simulation_revised_{condition}_imag.pdf'
        fig.savefig(filename,dpi=300)
        print(f"Figure saved as {filename}")

        fig, ax = plt.subplots(1,1,figsize=(2.5,1.6))
        c = ax.imshow(np.real(-masked_simPOIS).T*1e6, cmap=cmocean.cm.balance,extent=[np.min(x_lf),np.max(x_lf),np.min(y_lf),np.max(y_lf)],interpolation='nearest',origin='lower')
        ax.set_xlabel('$\it x$ (mm)')
        ax.set_ylabel('$\it y$ (mm)')
        ax.set_xlim(min(x_pois),max(x_pois))
        ax.set_ylim(min(y_pois),max(y_pois))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) 
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(colors='k', which='both', labelcolor='black')
        ax.set_aspect('equal', 'box')
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('$Re(\it V_{\mathrm{ae,LF}})$ ($\mu$V)',labelpad=3)
        ax.set_title('$\\nabla^2 \it V_\\mathrm{ae}=\it k_\\mathrm{ae} (\\nabla \it P \cdot \mathbf{E}_0)$',fontdict=fonttitle)
        fig.tight_layout()
        filename = f'./figures/quadrupole_simulation_poisson_{condition}_real.pdf'
        fig.savefig(filename,dpi=300)
        print(f"Figure saved as {filename}")

        fig, ax = plt.subplots(1,1,figsize=(2.5,1.6))
        c = ax.imshow(np.imag(-masked_simPOIS).T*1e6, cmap=cmocean.cm.balance,extent=[np.min(x_lf),np.max(x_lf),np.min(y_lf),np.max(y_lf)],interpolation='nearest',origin='lower')
        ax.set_xlabel('$\it x$ (mm)')
        ax.set_ylabel('$\it y$ (mm)')
        ax.set_xlim(min(x_pois),max(x_pois))
        ax.set_ylim(min(y_pois),max(y_pois))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) 
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(colors='k', which='both', labelcolor='black')
        ax.set_aspect('equal', 'box')
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('$Im(\it V_{\mathrm{ae,LF}})$ ($\mu$V)',labelpad=3)
        ax.set_title('$\\nabla^2 \it V_\\mathrm{ae}=\it k_\\mathrm{ae} (\\nabla \it P \cdot \mathbf{E}_0)$',fontdict=fonttitle)
        fig.tight_layout()
        filename = f'./figures/quadrupole_simulation_poisson_{condition}_imag.pdf'
        fig.savefig(filename,dpi=300)
        print(f"Figure saved as {filename}")

    df_quad_V = pd.read_csv('./simulations/quadrupole_potential_field.txt',sep='\s+',skiprows=9,names=['X','Y','V'],na_values=['NaN'])
    df_quad_V[['X','Y']] = df_quad_V[['X','Y']].round(2)
    df_quad_P = pd.read_csv('./simulations/quadrupole_pressure_field.txt',sep='\s+',skiprows=9,names=['X','Y','P'],na_values=['NaN'])
    df_quad_P['P'] = df_quad_P['P'].str.replace('i','j').apply(lambda x: np.complex128(x))
    df_quad_P[['X','Y']] = df_quad_P[['X','Y']].round(2)
    x_P,y_P = df_quad_P['X'].unique(),df_quad_P['Y'].unique()
    xn_P,yn_P = len(x_P),len(y_P)
    simQUADV = np.reshape(df_quad_V['V'].values,(xn_P,yn_P))
    simQUADP = np.reshape(df_quad_P['P'].values,(xn_P,yn_P))

    [gVx,gVy] = np.gradient(simQUADV,x_P,y_P)

    fig, ax = plt.subplots(1,1,figsize=(2.5,1.6))
    c = ax.imshow(simQUADV.T, cmap=cmocean.cm.balance,extent=[np.min(x_P),np.max(x_P),np.min(y_P),np.max(y_P)],interpolation='nearest',origin='lower')
    ax.streamplot(x_P, y_P, -gVx.T, -gVy.T, color='k', linewidth=0.2, arrowsize=0.4, density=0.4,broken_streamlines=True)
    ax.set_xlabel('$\it x$ (mm)')
    ax.set_ylabel('$\it y$ (mm)')
    ax.set_xlim(min(x_P),max(x_P))
    ax.set_ylim(min(y_P),max(y_P))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) 
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(colors='k', which='both', labelcolor='black')
    ax.set_aspect('equal', 'box')
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('$\it V$ (V)',labelpad=3)
    ax.set_title('Electric potential',fontdict=fonttitle)
    fig.tight_layout()
    filename = f'./figures/quadrupole_simulation_potential_field.pdf'
    fig.savefig(filename,dpi=300)
    print(f"Figure saved as {filename}")

    fig, ax = plt.subplots(1,1,figsize=(2.5,1.6))
    c = ax.imshow(np.real(-simQUADP).T/1e6, cmap=cmocean.cm.balance,alpha=1,vmin=-1,vmax=1,extent=[np.min(x_P),np.max(x_P),np.min(y_P),np.max(y_P)],interpolation='nearest',origin='lower')
    ax.set_xlabel('$\it x$ (mm)')
    ax.set_ylabel('$\it y$ (mm)')
    ax.set_xlim(min(x_P),max(x_P))
    ax.set_ylim(min(y_P),max(y_P))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) 
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(colors='k', which='both', labelcolor='black')
    ax.set_aspect('equal', 'box')
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('$\it P$ (MPa)',labelpad=3)
    ax.set_title('Pressure field',fontdict=fonttitle)
    fig.tight_layout()
    filename = f'./figures/quadrupole_simulation_pressure_field.pdf'
    fig.savefig(filename,dpi=300)
    print(f"Figure saved as {filename}")

if __name__ == "__main__":
    main()