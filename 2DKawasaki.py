import numpy as np
import matplotlib.pyplot as plt
import time

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


# Initialize 2D array of spins
K = 0.7
gamma = np.tanh(2*K)
num_spins = 256
num_updates = 100*num_spins*num_spins
state = 2*np.random.randint(2, size=(num_spins, num_spins))-1

idx = np.linspace(0, num_spins - 1, num_spins)

i = 0

plt.imshow(state, cmap='Greys',  interpolation='nearest')
plt.show()

t0 = time.time()

while(i < num_updates):
	row = int(np.random.choice(idx))
	col = int(np.random.choice(idx))
	direction = np.random.choice((1, 2))

	rate = calculate_rate(row, col, gamma, state, direction, num_spins)
	draw = np.random.uniform(0, 1)
	# print rate
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
	i = i + 1
	if(i % 10000 == 0):
		print i

t1 = time.time()
total = t1 - t0
print total

plt.imshow(state, cmap='Greys',  interpolation='nearest')
plt.show()



