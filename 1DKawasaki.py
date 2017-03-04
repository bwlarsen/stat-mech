import numpy as np
import matplotlib.pyplot as plt
import time


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


# Initialize 2D array of spins
K = 1000000
gamma = np.tanh(2*K)
num_spins = 256
num_updates = 1000*num_spins
state = 2*np.random.randint(2, size=(num_spins))-1

idx = np.linspace(0, num_spins - 1, num_spins)

i = 0

plt.plot(idx, state)
plt.show()

initial = np.sum(state)
print initial

t0 = time.time()

while(i < num_updates):
	row = int(np.random.choice(idx))

	rate = calculate_rate(row, gamma, state, num_spins)
	draw = np.random.uniform(0, 1)
	# print rate
	if(draw <= rate):
		state[row] = -state[row]
		if(row == num_spins - 1):
			state[0] = -state[0]
		else:
			state[row + 1] = - state[row+1]

	i = i + 1

t1 = time.time()
total = t1 - t0
print total

final = np.sum(state)
print final

plt.plot(idx, state)
plt.show()



