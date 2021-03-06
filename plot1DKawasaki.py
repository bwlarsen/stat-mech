### Code for plotting one realization to the 1D Ising model with Kawasaki dynamics

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit


def calculate_rate(x, temp, state, max_spin):
	s = float(state[x])
	if (x == 0):
		s_1 = float(state[max_spin-1])
	else:
		s_1 = float(state[x - 1])

	if (x == max_spin-1):
		s1 = float(state[0])
		s2 = float(state[1])
	elif(x == max_spin -2):
		s1 = float(state[max_spin -1])
		s2 = float(state[0])
	else:
		s1 = float(state[x + 1])
		s2 = float(state[x + 2])

	
	transition = 0.5*(1.0 - temp* 0.5 * (s_1*s + s1*s2)) * 0.5*(1.0 - s*s1)
	return transition

def func(x, A, b):
	return A*np.power(x, b) 


# Initialize 2D array of spins
K = 0
gamma = 0.99999 #np.tanh(2*K)
num_spins = 5000
num_updates = 50*num_spins
state = 2*np.random.randint(2, size=(num_spins))-1

idx = np.linspace(0, num_spins - 1, num_spins)

i = 0

# plt.plot(idx, state)
# plt.show()

corr_all = np.zeros(num_spins)
corr = np.zeros(num_updates)

initial = np.sum(state)
print initial

t0 = time.time()

while(i < num_updates):
	row = int(np.random.choice(idx))

	rate = calculate_rate(row, gamma, state, num_spins)
	draw = np.random.uniform(0, 1)
	# print rate
	if(draw < rate):
		if(row == num_spins - 1):
			# store = state[0]
			# state[0] = state[num_spins -1]
			# state[num_spins -1] = store
			state[0] = - state[0]
			state[num_spins -1] = - state[num_spins -1]
		else:
			# store = state[row]
			# state[row + 1] = state[row]
			# state[row] = store
			state[row] = - state[row]
			state[row + 1] = - state[row+1]

	

	shift_state = np.roll(state, 1)
	corr_all = state * shift_state
	corr[i] =  1.0/float(num_spins) * float(np.sum(corr_all))

	i = i + 1

print corr

t1 = time.time()
total = t1 - t0
print total

final = np.sum(state)
print final

rho = 0.5*(1.0 - corr)
rho_inv = 1/rho

t = np.linspace(0, 1/float(num_spins) * float(num_updates), num_updates)

popt, pcov = curve_fit(func, t, rho_inv, p0=(1.0, 0.4))
print popt

fit_func = popt[0] * np.power(t, popt[1]) 

plt.plot(t, rho_inv)
axes = plt.gca()
plt.plot(t, fit_func)
#axes.set_ylim([0,1])
plt.show()

# plt.plot(idx, state)
# plt.show()



