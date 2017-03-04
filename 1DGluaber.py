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
	transition = 0.5*(1.0 - float(state[x])* np.tanh(temp*nn_sum))
	return transition

def func(x, A, b):
	return A*np.power(x, b)


# Initialize 2D array of spins
K = 100000
num_spins = 100
num_updates = 10*num_spins
num_trials = 1000


idx = np.linspace(0, num_spins - 1, num_spins)
updates = np.linspace(0, num_updates - 1, num_updates)

j = 0

corr_all = np.zeros(num_spins)
corr = np.zeros(num_updates)
corr_avg = np.zeros(num_updates)


t0 = time.time()

while(j < num_trials):

	state = 2*np.random.randint(2, size=(num_spins))-1
	i = 0

	while(i < num_updates):
		row = int(np.random.choice(idx))

		rate = calculate_rate(row, K, state, num_spins)
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

corr_avg = 1.0/float(num_trials) * corr_avg

t1 = time.time()
total = t1 - t0
print total
rho = 0.5*(1.0 - corr_avg)
rho_inv = 1/rho

t = np.linspace(0, 1/float(num_spins) * float(num_updates), num_updates)

popt, pcov = curve_fit(func, t, rho_inv, p0=(1.0, 0.5))
print popt

fit_func = popt[0] * np.power(t, popt[1])

plt.plot(t, rho_inv)
plt.plot(t, fit_func)
axes = plt.gca()
#axes.set_ylim([0,1])
plt.show()



