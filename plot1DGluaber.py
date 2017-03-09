### Code for simulatin and plotting one realization of the 1D Ising-Glauber model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
	transition = 0.5*(1.0 - float(state[x])* temp*nn_sum*0.5)
	return transition

def func(x, A, b):
	return A*np.power(x, b)


# Initialize 2D array of spins
K = 100000
num_spins = 100
gamma = 1
num_updates = 10000*num_spins
num_trials = 1000


idx = np.linspace(0, num_spins - 1, num_spins)
updates = np.linspace(0, num_updates - 1, num_updates)

j = 0

corr_all = np.zeros(num_spins)
corr = np.zeros(num_updates)
corr_avg = np.zeros(num_updates)


t0 = time.time()



state = 2*np.random.randint(2, size=(num_spins))-1
i = 0

while(i < num_updates):
	row = int(np.random.choice(idx))

	if((i == 0) or (i == 100) or (i == 1000) or (i ==9999)):
		plt.plot(idx, state)
		plt.pause(0.0001)
		axes = plt.gca()
		axes.set_ylim([-1,1])
		plt.show()
		plt.pause(0.0001)

	rate = calculate_rate(row, gamma, state, num_spins)
	draw = np.random.uniform(0, 1)
	if(draw <= rate):
		state[row] = -state[row]
	
	

	i = i+1


plt.plot(idx, state)
plt.pause(0.0001)
axes = plt.gca()
axes.set_ylim([-1,1])
plt.show()
plt.pause(0.0001)


t1 = time.time()
total = t1 - t0
print total






