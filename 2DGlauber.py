import numpy as np
import matplotlib.pyplot as plt
import time

def calculate_rate(x, y, temp, state, max_spin):
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


# Initialize 2D array of spins
K = 100000
num_spins = 256
num_updates = 150*num_spins*num_spins
state = 2*np.random.randint(2, size=(num_spins,num_spins))-1

idx = np.linspace(0, num_spins - 1, num_spins)

# plt.imshow(state, cmap='Greys',  interpolation='nearest')
# plt.show()

i = 0

t0 = time.time()

while(i < num_updates):
	row = int(np.random.choice(idx))
	col = int(np.random.choice(idx))

	rate = calculate_rate(row, col, K, state, num_spins)
	draw = np.random.uniform(0, 1)
	if(draw <= rate):
		state[row,col] = -state[row,col]
	i = i + 1

t1 = time.time()
total = t1 - t0
print total

plt.imshow(state, cmap='Greys',  interpolation='nearest')
plt.show()




