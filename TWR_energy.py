# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 18:54:12 2024

@author: Hamish.S
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from atmospheric_model import air_model

#The following is an optional customization of the plots
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)



# Below are defintions of the magntiudes of all constants

# Define Global Constants
g = 9.81 # gravitational acceleration (m/s^2)
g_in = 0 # desired acceleration inside drop drone (m/s^2)

# Define Drop Drone Properties
m_0 = 60 # unloaded mass of drop drone (kg)
m_p = 15 # payload mass (kg)
C_D = 0.75 # drag coefficient ()
A = 0.126 # cross-sectional area of drop drone (m^2)

# Define Thruster Properties
N = 4 # number of thrusters
D = 0.2 # inlet diamater of each thruster (m)
P_a_max = 150000 # max input power of each thruster (W)

# Define Flight Profile Constants
h_1f = 1 # maximum height of phase 1 - checks (m)
t_check = 10 # time of checks at y_1 (s)
v_1 = 0.25 # constant velocity during phase 1 (m/s)
v_5 = 1.5 # constant velocity during phase 5 for safe landing (m/s)
t_test = 60 # max time of microgravity in phase 3 (s) <-- actual time will depend on TWR


# Below are definitions calculated from constants

# Drop Drone Properties
m = m_0 + m_p # total mass of drop drone (kg)
C_b = m / (C_D * A) # ballistic coefficient of drop drone

# Thruster Properties
A_disks = N * np.pi * (D / 2)**2  # total thruster disk area (m^2)
P_max = N * P_a_max # total max power of drop drone
#%%
def simulation(TWR, h_max):
    T_max = TWR * (m * g) # max trust # total max thrust of drone (N)
    
    # Define Global Event Functions

    # Event to stop ascent when h > h_max
    def max_height_event(t, y):
        h = y[1]
        return h - h_max  # Stop when h > h_max
    max_height_event.terminal = True
    max_height_event.direction = 1

    # Event to stop drop test if height falls below h = 0
    def ground_event(t, y):
        return y[1]  # Stop when h < 0
    ground_event.terminal = True
    ground_event.direction = -1
    
    
    # Phase 1 - Launch
    # Initial conditions for launch
    v0_launch = 0  # initial velocity (m/s)
    h0_launch = 0  # initial position (m)
    
    t_1 = int(h_1f / v_1) # time to get to h_1f (s)
    t_check_int = int(t_check)
    t_span_launch = (0, t_1 + t_check)  # max time span for launch (s)
    t_launch = np.linspace(t_span_launch[0], t_span_launch[1], (t_1 + t_check_int)+1) # time data points (s)
    
    v_launch = np.zeros(len(t_launch))
    h_launch = np.zeros(len(t_launch))
    T_launch = np.zeros(len(t_launch))
    D_launch = np.zeros(len(t_launch))
    a_launch = np.zeros(len(t_launch))
    P_launch = np.zeros(len(t_launch))
    P_prop_launch = np.zeros(len(t_launch))
    launch_energy = 0
    
    
    for t in range(0, t_1):
        v_launch[t] = v_1 # constant launch velocity
    
    for t in range(0, t_1):
        h_launch[t] = h0_launch + v_1 * t
    
    for t in range(t_1, t_1 + t_check_int+1):
        h_launch[t] = h_1f
    
    for t in range(0, t_1 + t_check_int+1):
        T_launch[t] = m * g # weight since a=0
    
    for t in range(0, t_1 + t_check_int+1):
        T = T_launch[t]
        v = v_launch[t]
        h = h_launch[t]
        _, _, rho = air_model(h)
        D_launch[t] = -np.sign(v) * 0.5 * rho * C_D * A * (v**2)
        W = m * g  # Weight of drop drone
        P_hover = (W**(3/2)) / ((2 * rho * A_disks)**(1/2))  # Power needed to hover
        P_climb = (T-W) * v # power needed to climb/move
        # drag power ignored since its small in this phase
        P_launch[t] = P_hover + P_climb
        P_prop_launch[t] = T * v # propulsive power
        launch_energy += P_launch[t] * 1 # t_step is defined as 1 s here
    
    # Phase 2 - Ascent
    phase = 2
    
    
    # Main Function that Defines the ODE
    def dynamics_ascent(t, y):
        v, h = y
        _, _, rho = air_model(h)
        #print("v:%.2f | h:%.2f | rho:%.2f" % (v, h, rho))
        
        W = m * g  # Weight of drop drone
        D = 0.5 * rho * C_D * A * (v**2) # Drag on drop drone
        
        T = T_max # assume max thrust for quick ascent
        
        
        # Define differential equations describing motion
        dvdt = (T - W - D) / m
        dhdt = v
        return [dvdt, dhdt]
    
    # Define Local Functions
    def power_ascent(t, h, v):
        _, _, rho = air_model(h)
        T = T_max
        W = m*g # weight
        D = (1/2) * rho * C_D * A * (v**2) # drag
        P_hover = np.abs(np.abs(m * g)**(3/2) / (np.sqrt(2 * rho * A_disks)))
        F_net = T - W
        P_climb = F_net * v
        P_drag = D * v
        return P_hover + P_climb + P_drag
    
    # Define Local Event Functions
    
    # Event to stop ascent if power exceeds P_max
    def ascent_max_power_event(t, y):
        h = y[1]
        _, _, rho = air_model(h)
        v = y[0]
        P = power_ascent(t, h, v)
        #print("power_max; P=%.2f | P_max=%.2f" % (P, P_max))
        return P - P_max  # Stop when P > P_max
    ascent_max_power_event.terminal = True
    ascent_max_power_event.direction = 1
    
    ascent_event_list = [max_height_event, ground_event, ascent_max_power_event] #, max_thrust_event, max_power_event
    
    
    # Initial conditions for ascent
    v0_ascent = v_launch[-1]  # initial velocity (m/s)
    h0_ascent = h_1f  # initial position (m)
    t_span_ascent = (0, 50)  # max time span for ascent (s)
    t_eval=np.linspace(t_span_ascent[0], t_span_ascent[1], 100) # evaluation times (s)
    y0_ascent = [v0_ascent, h0_ascent]  # initial state
    
    
    # Solve the differential equation for ascent
    sol_ascent = solve_ivp(dynamics_ascent, t_span_ascent, y0_ascent, t_eval=t_eval, events=ascent_event_list)

    
    event_times = sol_ascent.t_events

    
    # Extract time steps
    time_steps = np.diff(sol_ascent.t)
    # Extract v and h
    v_ascent = sol_ascent.y[0] 
    h_ascent = sol_ascent.y[1]
    
    
    # Calculate useful parameters based on solution
    
    # Calculate thrust power, and angular speed values during ascent
    T_ascent = np.zeros(len(sol_ascent.y[1]))
    D_ascent = np.zeros(len(sol_ascent.y[1]))
    a_ascent = np.zeros(len(sol_ascent.y[1]))
    P_ascent = np.zeros(len(sol_ascent.y[1]))
    P_prop_ascent = np.zeros(len(sol_ascent.y[1]))
    ascent_energy = 0
    W = m * g # weight of drop drone
    
    # Record thrust, power and energy
    for j in range(0, len(T_ascent)):
        h = sol_ascent.y[1][j]
        _, _, rho = air_model(h)
        v = sol_ascent.y[0][j]
        D = 0.5 * rho * C_D * A * (v**2)
        D_ascent[j] = -np.sign(v) * D # drag opposing velocity
        T_ascent[j] = T_max # total propuslive thrust
        a_ascent[j] = (T_max - W - D) / m
        
        P_ascent[j] = power_ascent(t, h, v)
        P_prop_ascent[j] = T_ascent[j] * v # propsulive power
        
        if j > 0:
            # Calculate energy for each time step
            ascent_energy += P_ascent[j] * time_steps[j-1]
    
    # Phase 3 - Drop Test
    phase = 3
    
    # Main Function that Defines the ODE
    def dynamics_drop_test(t, y):
        v, h = y
        _, _, rho = air_model(h)
        
        W = m * g  # Weight of drop drone
        D = 0.5 * rho * C_D * A * (v**2) # Drag on drop drone
        T = D  # Magnitude of thrust equal to drag
        
        dvdt = (-T - W + D) / m # downward thrust and weight, upward drag
        dhdt = v
        return [dvdt, dhdt]
    
    
    # Define Local Functions
    def power_drop_test(t, h, v):
        _, _, rho = air_model(h)
        D = (1/2) * rho * C_D * A * (v**2) # drag
        P_drag = np.abs(D * v)
        return P_drag
    
    # Define Local Events
    
    # Event to stop drop test if thrust exceeds T_max
    def drop_max_thrust_event(t, y):
        h = y[1]
        _, _, rho = air_model(h)
        v = y[0]
        T = 0.5 * rho * C_D * A * (v**2) # thrust equal to drag
        #print("h:%.2f | v:%.2f | T:%.2f | T_max:%.2f " % (h, v, T, T_max))
        return T - T_max  # Stop when T > T_max
    drop_max_thrust_event.terminal = True
    drop_max_thrust_event.direction = 1
    
    # Event to stop drop test if power exceeds P_max
    def drop_max_power_event(t, y):
        h = y[1]
        _, _, rho = air_model(h)
        v = y[0]
        P = power_drop_test(t, h, v)
        #print("power_max reached")
        return P - P_max  # Stop when P > P_max
    drop_max_power_event.terminal = True
    drop_max_power_event.direction = 1
    
    drop_test_event_list = [ground_event, drop_max_thrust_event, drop_max_power_event]
    
    
    # Initial conditions for drop test
    v0_drop_test = sol_ascent.y[0][-1]  # velocity at the end of ascent
    h0_drop_test = sol_ascent.y[1][-1]  # position at the end of ascent
    t_span_drop_test = (0, t_test)  # time span for drop_test (s)
    y0_drop_test = [v0_drop_test, h0_drop_test]  # initial state
    
    
    # Solve the differential equation for drop_test
    sol_drop_test = solve_ivp(dynamics_drop_test, t_span_drop_test, y0_drop_test, t_eval=np.linspace(t_span_drop_test[0], t_span_drop_test[1], 100), events=drop_test_event_list)

    
    # Extract event times and states
    event_times = sol_drop_test.t_events[0]
    event_states = sol_drop_test.y_events[0]
     
    event_times = sol_drop_test.t_events
 
    
    # Extract time steps
    time_steps = np.diff(sol_drop_test.t)
    # Extract v and h
    v_drop_test = sol_drop_test.y[0] 
    h_drop_test = sol_drop_test.y[1]
    
    # Calculate thrust, power and angular speed during drop test
    T_drop_test = np.zeros(len(sol_drop_test.y[1]))
    D_drop_test = np.zeros(len(sol_drop_test.y[1]))
    a_drop_test = np.zeros(len(sol_drop_test.y[1]))
    P_drop_test = np.zeros(len(sol_drop_test.y[1]))
    P_prop_drop_test = np.zeros(len(sol_drop_test.y[1]))
    drop_test_energy = 0
    
    # Record values
    for j in range(0, len(sol_drop_test.y[1])):
        h = sol_drop_test.y[1][j]
        _, _, rho = air_model(h)
        v = sol_drop_test.y[0][j]
        D_drop_test[j] = -np.sign(v) * 0.5 * rho * C_D * A * (v**2)
        T_drop_test[j] = np.sign(v) * 0.5 * rho * C_D * A * (v**2) # thrust equal to drag in direction of v
        a_drop_test[j] = -g
        P_drop_test[j] = power_drop_test(t, h, v)
        P_prop_drop_test[j] = T_drop_test[j] * v # propsulive power
        
        if j > 0:
            # Calculate energy for each time step
            drop_test_energy += P_drop_test[j] * time_steps[j-1]
            
    # Phase 4 - Deceleration
    phase = 4
    
    # Main Function that Defines the ODE
    def dynamics_deceleration(t, y):
        v, h = y
        _, _, rho = air_model(h)
        #print("v:%.2f | h:%.2f | rho:%.2f" % (v, h, rho))
        
        W = m * g  # Weight of drop drone
        D = 0.5 * rho * C_D * A * (v**2) # Drag on drop drone
        
        T = T_max
        #print("T:%.2f | P_prop:%.2f | v:%.2f | h:%.2f | rho:%.2f" % (T, prop_power(T, v, rho), v, h, rho))
        
        # Define differential equations describing motion
        dvdt = (T - W + D) / m # upward thrust and drag, downward weight
        dhdt = v
        return [dvdt, dhdt]
    
    
    # Define Local Functions
    def power_deceleration(t, h, v):
        _, _, rho = air_model(h)
        T = T_max
        W = m*g # weight
        D = (1/2) * rho * C_D * A * (v**2) # drag
        P_hover = np.abs(np.abs(m * g)**(3/2) / (np.sqrt(2 * rho * A_disks)))
        F_net = T - W
        P_climb = F_net * v
        P_drag = D * v
        #print("P_hover:%.2f W| P_climb:%.2f W| P_drag:%.2f W" % (P_hover, P_climb, P_drag))
        return np.abs(P_hover + P_climb + P_drag)
    
    # Define Local Events
    
    # Event to stop deceleration if velocity is v_land (decelerated enough)
    def landing_velocity_event(t, y):
        return y[0] - v_5  # Stop when v =< v_5 (safe landing speed)
    landing_velocity_event.terminal = True
    landing_velocity_event.direction = 1
    
    # Event to stop drop test if thrust exceeds T_max
    def decel_max_thrust_event(t, y):
        h = y[1]
        _, _, rho = air_model(h)
        v = y[0]
        T = 0.5 * rho * C_D * A * (v**2) # thrust equal to drag
        return T - T_max  # Stop when T > T_max
    decel_max_thrust_event.terminal = True
    decel_max_thrust_event.direction = 1
    
    # Event to stop deceleration if power exceeds P_max
    def decel_max_power_event(t, y):
        h = y[1]
        _, _, rho = air_model(h)
        v = y[0]
        P = power_deceleration(t, h, v)
        #print("power_max reached phase %d; P=%.2f" % (phase, P))
        return P - P_max  # Stop when P > P_max
    decel_max_power_event.terminal = True
    decel_max_power_event.direction = 1
    
    event_list = [landing_velocity_event, ground_event, decel_max_thrust_event, decel_max_power_event]
    
    
    # Initial conditions for landing
    v0_deceleration = sol_drop_test.y[0][-1]  # velocity at the end of drop_test
    h0_deceleration = sol_drop_test.y[1][-1]  # position at the end of drop_test
    t_span_deceleration = (0, 50)  # time span for landing (s)
    y0_deceleration = [v0_deceleration, h0_deceleration]  # initial state
    
    # Solve the differential equation for deceleration
    sol_deceleration = solve_ivp(dynamics_deceleration, t_span_deceleration, y0_deceleration, t_eval=np.linspace(t_span_deceleration[0], t_span_deceleration[1], 100), events=event_list)

    
    # Extract event times and states
    event_times = sol_deceleration.t_events[0]
    event_states = sol_deceleration.y_events[0]
    
    event_times = sol_deceleration.t_events
    
    # Extract time steps
    time_steps = np.diff(sol_drop_test.t)
    # Extract v and h
    v_deceleration = sol_deceleration.y[0] 
    h_deceleration = sol_deceleration.y[1]
    
    # Calculate Useful Parameters from sol_deceleration
    T_deceleration = np.zeros(len(sol_deceleration.y[1]))
    D_deceleration = np.zeros(len(sol_deceleration.y[1]))
    a_deceleration = np.zeros(len(sol_deceleration.y[1]))
    P_deceleration = np.zeros(len(sol_deceleration.y[1]))
    P_prop_deceleration = np.zeros(len(sol_deceleration.y[1]))
    deceleration_energy = 0
    
    # Record thrust and power values during deceleration
    for j in range(0, len(sol_deceleration.y[1])):
        h = sol_deceleration.y[1][j]
        _, _, rho = air_model(h)
        v = sol_deceleration.y[0][j]
        T_deceleration[j] = T_max
        D = 0.5 * rho * C_D * A * (v**2)
        D_deceleration[j] = -np.sign(v) * D # drag opposing velocity
        a_deceleration[j] = (T_max - W + D) / m
        
        P_deceleration[j] = power_deceleration(t, h, v)
        P_prop_deceleration[j] = T_deceleration[j] * v # propsulive power
        
        if j > 0:
            # Calculate energy for each time step
            deceleration_energy += P_deceleration[j] * time_steps[j-1]
    
    # Phase 5 - Landing
    # Initial conditions for landing
    v0_landing = v_deceleration[-1]  # initial velocity (m/s)
    h0_landing = h_deceleration[-1]  # initial position (m)
    
    t_5 = int(h0_landing / v_5) # time to get to ground (s)
    t_span_landing = (0, t_5)  # max time span for landing (s)
    t_landing = np.linspace(t_span_landing[0], t_span_landing[1], t_5+1) # time data points (s)
    
    v_landing = np.zeros(len(t_landing))
    h_landing = np.zeros(len(t_landing))
    T_landing = np.zeros(len(t_landing))
    D_landing = np.zeros(len(t_landing))
    a_landing = np.zeros(len(t_landing))
    P_landing = np.zeros(len(t_landing))
    P_prop_landing = np.zeros(len(t_landing))
    landing_energy = 0
    
    for t in range(0, t_5+1):
        v_landing[t] = -v_5 # constant descending landing velocity
    
    for t in range(0, t_5+1):
        h_landing[t] = h0_landing - v_5 * t
    
    for t in range(0, t_5+1):
        T_landing[t] = m * g # weight since a=0
    
    for t in range(0, t_5+1):
        T = T_landing[t]
        v = v_landing[t]
        h = h_landing[t]
        _, _, rho = air_model(h)
        D_landing[t] = -np.sign(v) * 0.5 * rho * C_D * A * (v**2)
        W = m * g  # Weight of drop drone
        P_hover = (W**(3/2)) / ((2 * rho * A_disks)**(1/2))  # Power needed to hover
        P_climb = (T-W) * v # power needed to climb/move
        # drag power ignored since its small in this phase
        P_landing[t] = P_hover + P_climb
        P_prop_landing[t] = T * v # propsulive power
        landing_energy += P_landing[t] * 1 # each time step is 1 s here

    return (launch_energy + ascent_energy + drop_test_energy + deceleration_energy + landing_energy) / 1e6

#%%
#print("Drop test time = %.2f s" % simulation(1.7))
TWR_min = 1.05
TWR_max = 5
h_max_values = [750, 1000, 1700, 2500]

TWR_values = np.linspace(TWR_min, TWR_max, 100)
energy_values = np.zeros((len(h_max_values), len(TWR_values)))

for j in range(0, len(h_max_values)):
    for i in range(0, len(TWR_values)):
        TWR = TWR_values[i]
        h_max = h_max_values[j]
        energy_values[j][i] = simulation(TWR, h_max)
plt.figure(figsize=(12, 6))
axis_label_fontsize = 20
plt.subplot(1, 1, 1)

for plot_num in range(0, len(h_max_values)):
    h_max = h_max_values[plot_num]
    plt.plot(TWR_values, energy_values[plot_num][:], label=f'{h_max/1000} km')

plt.xlabel('Thrust to Weight Ratio, TWR',fontsize=axis_label_fontsize)
plt.ylabel('Energy Required (MJ)',fontsize=axis_label_fontsize)
plt.grid(True)
plt.legend(fontsize=16)
plt.tight_layout()
#%%
# TWR_min = 1.05
# TWR_max = 5

# TWR_values = np.linspace(TWR_min, TWR_max, 100)
# energy_values = np.zeros(len(TWR_values))


# for i in range(0, len(TWR_values)):
#     TWR = TWR_values[i]
#     h_max = 1700
#     energy_values[i] = simulation(TWR, h_max)
# #
# # Plot the results
# plt.figure(figsize=(10, 5))
# axis_label_fontsize = 20
# plt.subplot(1, 1, 1)

# plt.plot(TWR_values, energy_values)

# plt.xlabel('Thrust to Weight Ratio, TWR',fontsize=14)
# plt.ylabel('Energy Requiured (MJ)',fontsize=14)
# plt.grid(True)
# plt.tight_layout()