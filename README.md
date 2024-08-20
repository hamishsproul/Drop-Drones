# Drop-Drones

These python files were used to analyze the dynamics of a drone-based low gravity platform for my thesis titled Drop Drones: An Altenative Low Gravity Platform. Feel free to use them as you wish.


## drop_drone_solve_ivp.py

This file utlizes the SciPy routine solve_ivp to model the flight profile of a drop drone using defined performance characteristics. This is the main file used in my thesis.


## atmospheric_model.py

This file implements the US Standard Atmospheric Model. It is required by drop_drone_solve_ivp.


## TWR_test_time.py

This file plots the duration of low gravity against the thrust to weight ratio.


## TWR_energy.py

This file plots the energy required for a flight profile against the thrust to weight ratio.
