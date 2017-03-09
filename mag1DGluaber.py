### Code for simulating the decay of the magnetization in the 1D Ising-Glauber model

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit

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

def func(x, A, b):
	return A*np.exp(b*x)


# Initialize 2D array of spins
K = 100000
gamma = 1
num_spins = 100
num_updates = 200*num_spins
num_trials = 500


idx = np.linspace(0, num_spins - 1, num_spins)
updates = np.linspace(0, num_updates - 1, num_updates)

j = 0


mag = np.zeros(num_updates)
mag_avg = np.zeros(num_updates)


t0 = time.time()

while(j < num_trials):

	#state = 2*np.random.randint(2, size=(num_spins))-1
	state = np.ones(num_spins)
	i = 0

	while(i < num_updates):
		row = int(np.random.choice(idx))

		rate = calculate_rate(row, gamma, state, num_spins)
		draw = np.random.uniform(0, 1)
		if(draw <= rate):
			state[row] = -state[row]
		
		
		mag[i] =  1.0/float(num_spins) * float(np.sum(state))

		i = i+1

	mag_avg = mag_avg + mag
	j = j+1
	if(j % 50 == 0):
		print j

mag_avg = 1.0/float(num_trials) * mag_avg

t1 = time.time()
total = t1 - t0
print total

t = np.linspace(0, 1/float(num_spins) * float(num_updates), num_updates)

#popt, pcov = curve_fit(func, t, mag_avg, p0=(1.0, -0.5))
#print popt

fit_func = np.exp(-(1 - gamma)*t)

sim = plt.plot(t, mag_avg, label = 'Simulation')
theory = plt.plot(t, fit_func, label = 'Theory')
plt.xlabel('t')
plt.ylabel('m')

plt.legend(numpoints = 3)

axes = plt.gca()
axes.set_ylim([0,1.25])
plt.show()



