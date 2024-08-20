# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:19:39 2024

@author: Hamish
"""
import os
import numpy as np
import matplotlib.pyplot as plt

#The following is an optional customization of the plots
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)

plot_graphs = False

#%%
def air_model(h):
    """
    This function estimates the air density, pressure and temperature based on the current altitude.
    The atmosphere is split into 4 layers defined by the US Standard Atmopsheric Model.
    The fourth (final) layer is valid up to 47 km.
    """
    
    #Define local constants
    M = 0.02897 # avg molar mass of atmosphere (kg/mol)
    g = 9.81 # gravitational acceleration (m/s^2)
    R = 8.3145 # gas constant (J/mol K)

    """
    Define layer-specific constants
    
    columns:
    0: initial altitude of layer (m)
    1: temperature lapse rate (K/m)
    2: initial density of layer (kg/m^3)
    3: initial temperature of layer (K)
    4: initial pressure of layer (Pa)
    """
    air_param = [
        [00000, -0.0065, 1.22500, 288.20, 101325.00],
        [11000, +0.0000, 0.36383, 216.65, 022632.10],
        [20000, +0.0010, 0.08801, 216.65, 005474.89],
        [32000, +0.0028, 0.01322, 228.65, 000868.02]
        ]
    
    "Calculate air density rho, air pressure P, and air temperature T based on height and model layer"
    #layer 0 (0-11km)
    if h >= air_param[0][0] and h <= air_param[1][0]: 
        T = air_param[0][3] + air_param[0][1] * (h - air_param[0][0]) # temperature
        exponent = -(g * M) / (R * air_param[0][1]) # the general exponent in the following fns
        P = air_param[0][4] * ((T / air_param[0][3])**(exponent)) # air pressure 
        rho = air_param[0][2] * ((T / air_param[0][3])**(exponent - 1)) # air density
        #p = air_param[0][4] * (1 - (air_param[0][1] * h) / air_param[0][3])**((g * M) / (R * air_param[0][1]))
        
    #layer 1 (11-20km)    
    elif h > air_param[1][0] and h <= air_param[2][0]:
        T = air_param[1][3] # temperature is constant (isothermal layer)
        exponent = -(g * M * (h - air_param[1][0])) / (R * T) # the exponent in exp(x)
        P = air_param[1][4] * np.exp(exponent) # air pressure
        rho = air_param[1][2] * np.exp(exponent) # air density 
    
    #layer 2 (20-32km)                           
    elif h > air_param[2][0] and h <= air_param[3][0]:
        T = air_param[2][3] + air_param[2][1] * (h - air_param[2][0]) # temperature 
        exponent = -(g * M) / (R * air_param[2][1]) # the general exponent in the following fns
        P = air_param[2][4] * ((T / air_param[2][3])**(exponent)) # air pressure 
        rho = air_param[2][2] * ((T / air_param[2][3])**(exponent - 1)) # air density
        
    #layer 3 (32-47km)
    elif h > air_param[3][0] and h <= 47000: 
        T = air_param[3][3] + air_param[3][1] * (h - air_param[3][0]) # temperature 
        exponent = -(g * M) / (R * air_param[3][1]) # the general exponent in the following fns
        P = air_param[3][4] * ((T / air_param[3][3])**(exponent)) # air pressure 
        rho = air_param[3][2] * ((T / air_param[3][3])**(exponent - 1)) # air density
        
    else:
        T = 1e-5
        P = 1e-5
        rho = 1e-5
        #print("error with altitude in air_model")
    
    return T, P, rho

#%%
# Calculate T, P and rho
h = np.linspace(0, 47000, 500)

T = np.zeros_like(h)
P = np.zeros_like(h)
rho = np.zeros_like(h)

for i in range(0, len(h)):
    T[i], P[i], rho[i] = air_model(h[i])

#%%
# Plot the results
if plot_graphs == True:
    plt.close("all")
    fig, axs = plt.subplots(figsize=(6, 6)) # This creates the figure
    
    # Create subplots    
    fig, axs = plt.subplots(1, 3, figsize=(10, 6))  # subplots horizontally aligned
    
    # Temperature plot
    axs[0].plot(T, h / 1000, color = "orange", label="Temperature")
    axs[0].set_xlabel('Temperature (K)',fontsize=18)
    axs[0].grid(True)
    
    # Pressure Plot
    axs[1].plot(P / 1000, h / 1000, color = "blue", label="Pressure")
    axs[1].set_xlabel('Air Pressure (kPa)',fontsize=18)
    axs[1].grid(True)
    
    # Air Density plot
    axs[2].plot(rho, h / 1000, color = "green", label="Air Density")
    axs[2].set_xlabel('Air Density ($\mathrm{kg/m^{3}}$)',fontsize=18)
    axs[2].grid(True)
    
    
    
    fig.supylabel('Altitude (km)',fontsize=18)
    #fig.suptitle("US Standard Atmospheric Model", fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    Name=f'Atmospheric Model.png'
    folder_path = 'Python_Plots'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plot_path = os.path.join(folder_path, Name)
    plt.savefig(plot_path, dpi=200)  
