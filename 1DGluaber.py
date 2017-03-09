### Code for simulating the dynamics of the 1D Glauber model

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit

# Function for calculating transition rates
def calculate_rate(x, temp, state, max_spin):
	if (x == 0):
		s1 = state[max_spin-1]
	else:
		s1 = state[x - 1]

	if (x == max_spin-1):
		s2 = state[0]
	else:
		s2 = state[x+1]

	nn_sum = s1 + s2
	transition = 0.5*(1.0 - float(state[x])* temp*0.5*float(nn_sum))
	return transition

def func(x, A, b, c):
	return A*np.power(x, b) + c


# Initialize 1D array of spins
K = 100000
gamma = 1
num_spins = 100
num_updates = 100*num_spins
num_trials = 1000


idx = np.linspace(0, num_spins - 1, num_spins)
updates = np.linspace(0, num_updates - 1, num_updates)

j = 0

corr_all = np.zeros(num_spins)
corr = np.zeros(num_updates)
corr_avg = np.zeros(num_updates)


t0 = time.time()

while(j < num_trials):

	# Randomly initialize the spins each trial
	state = 2*np.random.randint(2, size=(num_spins))-1
	i = 0

	while(i < num_updates):
		row = int(np.random.choice(idx))

		rate = calculate_rate(row, gamma, state, num_spins)
		draw = np.random.uniform(0, 1)
		if(draw <= rate):
			state[row] = -state[row]
		
		shift_state = np.roll(state, 1)
		corr_all = state * shift_state
		corr[i] =  1.0/float(num_spins) * float(np.sum(corr_all))

		i = i+1

	corr_avg = corr_avg + corr
	j = j+1
	if(j % 50 == 0):
		print j

# corr_avg stores the average correlation at each update
corr_avg = 1.0/float(num_trials) * corr_avg

# Fit to the power law for the mean domain wall separation
t1 = time.time()
total = t1 - t0
print total
rho = 0.5*(1.0 - corr_avg)
rho_inv = 1/rho

t = np.linspace(0, 1/float(num_spins) * float(num_updates), num_updates)

popt, pcov = curve_fit(func, t[10:num_updates - 1], rho_inv[10:num_updates - 1], p0=(1.0, 0.5, 0))
print popt

fit_func = popt[0] * np.power(t, popt[1]) + popt[2]

sim = plt.plot(t, rho_inv, label = 'Simulation')
theory = plt.plot(t, fit_func, label = 'Fit')
axes = plt.gca()
#axes.set_ylim([0,1])
plt.legend(numpoints = 3, loc = 'upper left')
plt.show()



