import numpy as np
import matplotlib.pyplot as plt
import time

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


# Initialize 2D array of spins
K = 100000
num_spins = 1000
num_updates = 500*num_spins
state = 2*np.random.randint(2, size=(num_spins))-1

idx = np.linspace(0, num_spins - 1, num_spins)

i = 0



t0 = time.time()

while(i < num_updates):
	row = int(np.random.choice(idx))

	rate = calculate_rate(row, K, state, num_spins)
	draw = np.random.uniform(0, 1)
	if(draw <= rate):
		state[row] = -state[row]
	i = i + 1

t1 = time.time()
total = t1 - t0
print total

plt.plot(idx, state)
plt.show()



