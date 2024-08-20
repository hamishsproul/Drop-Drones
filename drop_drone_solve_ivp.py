# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:02:21 2024

@author: Hamish

This script implements the work of Kedarisetty and Manathara (2023) and Snyder et al (2022) 
to model the flight profile of a propellor-driven drop drone.

Bugs probably exist. Think critically about the resulting flight profile.
A succesful flight profile will probably see event 1 in phase 2 (max height), event 2 in 
phase 3 (max thrust), and event 1 in phase 4 (landing velocity achieved).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from atmospheric_model import air_model

#The following is an optional customization of the plots
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)



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
N = 5 # number of thrusters
TWR = 2.68 # thrust-to-weight ratio
rev_thrust = 0.5 # 50% since reverse thrust not as effective
D = 0.2 # inlet diamater of each thruster (m)
P_a_max = 59000 # max input power of each thruster (W)

# Define Flight Profile Constants
h_max = 750 # maximum height of ascent phase (m)
h_1f = 1 # maximum height of phase 1 - checks (m)
t_check = 10 # time of checks at y_1 (s)
v_1 = 0.25 # constant velocity during phase 1 (m/s)
v_5 = 1.5 # constant velocity during phase 5 for safe landing (m/s)
t_test = 25 # max time of microgravity in phase 3 (s) <-- actual time will depend on TWR


# Below are definitions calculated from constants

# Drop Drone Properties
m = m_0 + m_p # total mass of drop drone (kg)
C_b = m / (C_D * A) # ballistic coefficient of drop drone

# Thruster Properties
T_max = TWR * (m * g) # max trust # total max thrust of drone (N)
A_disks = N * np.pi * (D / 2)**2  # total thruster disk area (m^2)
P_max = N * P_a_max # total max power of drop drone
#%%
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
#%%
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
#%%
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
print("Iterations by ODE solver in phase %d: %d" % (phase, sol_ascent.nfev))

event_times = sol_ascent.t_events
for i, event_time in enumerate(event_times):
    if event_time.size > 0:
        print("Event %d phase %d occurred at times: %.2f s" % (i+1, phase, event_time))

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
#%%
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
    return T - rev_thrust * T_max  # Stop when T > rev_thrust * T_max
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
print("Iterations by ODE solver in phase %d: %d" % (phase, sol_drop_test.nfev))

# Extract event times and states
event_times = sol_drop_test.t_events[0]
event_states = sol_drop_test.y_events[0]
 
event_times = sol_drop_test.t_events
for i, event_time in enumerate(event_times):
    if event_time.size > 0:
        print("Event %d phase %d occurred at times: %.2f s" % (i+1, phase, event_time))

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
#%%
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
print("Iterations by ODE solver in phase %d: %d" % (phase, sol_deceleration.nfev))

# Extract event times and states
event_times = sol_deceleration.t_events[0]
event_states = sol_deceleration.y_events[0]

event_times = sol_deceleration.t_events
for i, event_time in enumerate(event_times):
    if event_time.size > 0:
        print("Event %d phase %d occurred at times: %.2f s" % (i+1, phase, event_time))

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
#%%
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
#%%
# Combine data from all phases

t_combined = np.concatenate([t_launch, sol_ascent.t + t_launch[-1], sol_drop_test.t + sol_ascent.t[-1] + t_launch[-1], sol_deceleration.t + t_launch[-1] + sol_ascent.t[-1] + sol_drop_test.t[-1], t_landing + t_launch[-1] + sol_ascent.t[-1] + sol_drop_test.t[-1] + sol_deceleration.t[-1]])
v_combined = np.concatenate([v_launch, sol_ascent.y[0], sol_drop_test.y[0], sol_deceleration.y[0], v_landing])
h_combined = np.concatenate([h_launch, sol_ascent.y[1], sol_drop_test.y[1], sol_deceleration.y[1], h_landing])
T_combined = np.concatenate([T_launch, T_ascent, T_drop_test, T_deceleration, T_landing])
D_combined = np.concatenate([D_launch, D_ascent, D_drop_test, D_deceleration, D_landing])
a_combined = np.concatenate([a_launch, a_ascent, a_drop_test, a_deceleration, a_landing])
P_combined = np.concatenate([P_launch, P_ascent, P_drop_test, P_deceleration, P_landing])
P_prop_combined = np.concatenate([P_prop_launch, P_prop_ascent, P_prop_drop_test, P_prop_deceleration, P_prop_landing])
energy_combined = launch_energy + ascent_energy + drop_test_energy + deceleration_energy + landing_energy

drop_test_time = sol_drop_test.t[-1]
print("Total energy = %.2f MJ" % (energy_combined / 1e6))
print("Drop test time = %.2f s" % drop_test_time)
print("ascent: %.2f s | deceleration: %.2f s" % (sol_ascent.t[-1], sol_deceleration.t[-1]))
#%%
#Plot Graphs
plot_num = 5
plt.figure(figsize=(15, plot_num*3))
axis_label_fontsize = 20

plt.subplot(plot_num, 1, 1)
plt.plot(t_combined, h_combined, color='blue')
#plt.xlabel('Time (s)', fontsize=axis_label_fontsize)
plt.ylabel('Position (m)', fontsize=axis_label_fontsize)
plt.grid()

plt.subplot(plot_num, 1, 2)
plt.plot(t_combined, v_combined, color='green')
#plt.xlabel('Time (s)', fontsize=axis_label_fontsize)
plt.ylabel('Velocity (m/s)',fontsize=axis_label_fontsize)
plt.grid()

plt.subplot(plot_num, 1, 3)
plt.plot(t_combined, a_combined, color='red')
#plt.xlabel('Time (s)', fontsize=axis_label_fontsize)
plt.ylabel('Acceleration ($\mathrm{m/s^{2}}$)', fontsize=axis_label_fontsize)
plt.grid()

plt.subplot(plot_num, 1, 4)
plt.plot(t_combined, T_combined, color='orange')
#plt.xlabel('Time (s)', fontsize=axis_label_fontsize)
plt.ylabel('Thrust (N)', fontsize=axis_label_fontsize)
plt.grid()

# plt.subplot(plot_num, 1, 4)
# plt.plot(t_combined, D_combined, color='red')
# plt.xlabel('Time (s)', fontsize=axis_label_fontsize)
# plt.ylabel('Drag (N)', fontsize=axis_label_fontsize)
# plt.grid()

plt.subplot(plot_num, 1, 5)
plt.plot(t_combined, P_combined / 1000, color='purple')
plt.xlabel('Time (s)', fontsize=axis_label_fontsize)
plt.ylabel('Power (kW)', fontsize=axis_label_fontsize)
plt.grid()
plt.tight_layout()

# # Save the plot
# Name=f'drop_drone_fsolve_h_max_{h_max}_T_max_{T_max: .3f}_C_D_{C_D}_A_{A}_m_{m}_t_{drop_test_time: .3f}_E_{energy_combined/1000000: .3f}.png'
# folder_path = 'Python_Plots'
# if not os.path.exists(folder_path):
#     os.makedirs(folder_path)
# plot_path = os.path.join(folder_path, Name)
# plt.savefig(plot_path, dpi=200)

plt.show()
 #%%
# # Plot Graphs
# plt.figure(figsize=(10, 12))
# axis_label_fontsize = 14

# plt.subplot(6, 1, 1)
# plt.plot(t_combined, h_combined, color='blue')
# plt.xlabel('Time (s)', fontsize=axis_label_fontsize)
# plt.ylabel('Position (m)', fontsize=axis_label_fontsize)
# plt.grid()

# plt.subplot(6, 1, 2)
# plt.plot(t_combined, v_combined, color='green')
# plt.xlabel('Time (s)', fontsize=axis_label_fontsize)
# plt.ylabel('Velocity (m/s)',fontsize=axis_label_fontsize)
# plt.grid()

# plt.subplot(6, 1, 3)
# plt.plot(t_combined, T_combined, color='orange')
# plt.xlabel('Time (s)', fontsize=axis_label_fontsize)
# plt.ylabel('Thrust (N)', fontsize=axis_label_fontsize)
# plt.grid()

# plt.subplot(6, 1, 4)
# plt.plot(t_combined, D_combined, color='red')
# plt.xlabel('Time (s)', fontsize=axis_label_fontsize)
# plt.ylabel('Drag (N)', fontsize=axis_label_fontsize)
# plt.grid()

# plt.subplot(6, 1, 5)
# plt.plot(t_combined, P_combined / 1000, color='purple')
# plt.xlabel('Time (s)', fontsize=axis_label_fontsize)
# plt.ylabel('Power (kW)', fontsize=axis_label_fontsize)
# plt.grid()

# plt.subplot(6, 1, 6)
# plt.plot(t_combined, np.abs(P_prop_combined) / 1000, color='purple')
# plt.xlabel('Time (s)', fontsize=axis_label_fontsize)
# plt.ylabel('Propulsive Power (kW)', fontsize=axis_label_fontsize)
# plt.grid()

# plt.tight_layout()
# plt.show()

