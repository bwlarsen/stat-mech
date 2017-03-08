import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit

def calculate_rate(x, y, temp, state, dir, max_spin):
	s = float(state[x, y])
	if(dir == 1):
		if (x == 0):
			s_1 = float(state[max_spin-1, y])
		else:
			s_1 = float(state[x - 1, y])

		if (x == max_spin-1):
			s1 = float(state[0, y])
			s2 = float(state[1, y])
		elif(x == max_spin -2):
			s1 = float(state[max_spin -1, y])
			s2 = float(state[0, y])
		else:
			s1 = float(state[x + 1, y])
			s2 = float(state[x + 2, y])

	if(dir == 2):
		if (y == 0):
			s_1 = float(state[x, max_spin-1])
		else:
			s_1 = float(state[x, y - 1])

		if (y == max_spin-1):
			s1 = float(state[x, 0])
			s2 = float(state[x, 1])
		elif(y == max_spin -2):
			s1 = float(state[x, max_spin -1])
			s2 = float(state[x, 0])
		else:
			s1 = float(state[x, y + 1])
			s2 = float(state[x, y + 2])

	
	transition = 0.5*(1.0 - temp* 0.5 * (s_1*s + s1*s2)) * 0.5*(1.0 - s*s1)
	return transition


def func(x, A, b, c):
	return A*np.power(x, b) + c

####

K = 2*0.441
gamma = np.tanh(2*K)
num_spins = 25
num_updates = 75*num_spins*num_spins
#state = 2*np.random.randint(2, size=(num_spins,num_spins))-1

idx = np.linspace(0, num_spins - 1, num_spins)

# plt.imshow(state, cmap='Greys',  interpolation='nearest')
# plt.show()

corr_all = np.zeros(num_spins)
corr = np.zeros(num_updates)
corr_avg = np.zeros(num_updates)

t0 = time.time()
 
num_trials = 10

g1 = np.zeros([num_trials,num_updates]) 

j = 0
 # Perform the updates
while(j < num_trials):
	state = 2*np.random.randint(2, size=(num_spins,num_spins))-1
	#g1 = np.zeros([num_trials,num_updates]) 

	for i in range(num_updates):
		row = int(np.random.choice(idx))
		col = int(np.random.choice(idx))
		direction = np.random.choice((1, 2))

		rate = calculate_rate(row, col, gamma, state, direction, num_spins)
		draw = np.random.uniform(0, 1)

		if(draw <= rate):
			state[row, col] = -state[row, col]
			if (direction == 1):
				if(row == num_spins - 1):
					state[0, col] = -state[0, col]
				else:
					state[row + 1, col] = - state[row+1, col]
			if (direction == 2):
				if(col == num_spins - 1):
					state[row, 0] = -state[row, 0]
				else:
					state[row, col + 1] = - state[row, col + 1]
		
		for k in range(10):
			shift_state = np.roll(state[k+15,:], 1)
			corr_all = state[k+15,:] * shift_state     
			corr[i] =  corr[i] + 1.0/float(num_spins) * float(np.sum(corr_all))
		corr[i] = 0.1 * corr[i]
		
	corr_avg = corr_avg + corr
	j = j + 1
	print(j)

corr_avg = 1.0/float(num_trials) * corr_avg
print corr_avg
rho = 0.5*(1.0 - corr_avg)
rho_inv = 1/rho

t1 = time.time()
total = t1 - t0
print('total')

print rho_inv

t = np.linspace(0, 1/float(num_spins*num_spins) * float(num_updates), num_updates)

popt, pcov = curve_fit(func, t[20:num_updates - 1], rho_inv[20:num_updates - 1], p0=(1.0, 0.4, 0))
print popt

fit_func = popt[0] * np.power(t, popt[1]) + popt[2]


sim = plt.plot(t, rho_inv, label = 'Simulation')
theory = plt.plot(t, fit_func, label = 'Fit')
axes = plt.gca()
#axes.set_ylim([0,1])
plt.legend(numpoints = 3, loc = 'upper left')
plt.show()

