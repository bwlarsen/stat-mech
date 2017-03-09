#### Code for calculating the dynamics of the 2D Glauber model

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit

# Function to calculate transition rates
def calculate_rate(x, y, temp, state, max_spin):

	#This is to account for periodic boundary conditions
	if (x == 0):
		s1 = state[max_spin-1, y]
	else:
		s1 = state[x - 1, y]

	if (y == 0):
		s2 = state[x, max_spin-1]
	else:
		s2 = state[x , y-1]

	if (x == max_spin-1):
		s3 = state[0, y]
	else:
		s3 = state[x+1 , y]

	if (y == max_spin-1):
		s4 = state[x, 0]
	else:
		s4 = state[x , y+1]

	nn_sum = s1 + s2 + s3 + s4
	transition = 0.5*(1.0 - float(state[x,y])* np.tanh(temp*nn_sum))
	return transition

def func(x, A, b, c):
	return A*np.power(x, b) + c


# Initialize 2D array of spins
K = 0.6
num_spins = 30
num_updates = 50*num_spins*num_spins


idx = np.linspace(0, num_spins - 1, num_spins)



i = 0

t0 = time.time()
 
num_trials = 200

g1 = np.zeros([num_trials,num_updates]) 

j = 0
 # Perform the trials
while(j < num_trials):
    state = 2*np.random.randint(2, size=(num_spins,num_spins))-1
    #g1 = np.zeros([num_trials,num_updates]) 
    #Perform the updates
    for i in range(num_updates):
        row = int(np.random.choice(idx))
        col = int(np.random.choice(idx))
    
        rate = calculate_rate(row, col, K, state, num_spins)
        draw = np.random.uniform(0, 1)
     
        if(draw <= rate):
             state[row,col] = -state[row,col]
        s = 0     
        for k in range(num_spins-1):
            
            #For each update, calculate the corellation between NN
            if(k == num_spins-1):
                s = s + state[15,k]*state[15,1]
            else: 
                s = s + state[15,k]*state[15,k+1]
            g1[j,i] = (1/num_spins)*s      
    j = j + 1
    print(j)

# Calculate the domain wall density
rho = (0.5)*(1.0 - np.mean(g1,axis=0))
rho_inv = 1/rho

t1 = time.time()
total = t1 - t0
print('total')

t = np.linspace(0, 1/float(num_spins) * float(num_updates), num_updates)

popt, pcov = curve_fit(func, t[10:num_updates - 1], rho_inv[10:num_updates - 1], p0=(1.0, 0.4, 0))
print popt

fit_func = popt[0] * np.power(t, popt[1]) + popt[2]


sim = plt.plot(t, rho_inv, label = 'Simulation')
theory = plt.plot(t, fit_func, label = 'Fit')
axes = plt.gca()
#axes.set_ylim([0,1])
plt.legend(numpoints = 3, loc = 'upper left')
plt.show()




