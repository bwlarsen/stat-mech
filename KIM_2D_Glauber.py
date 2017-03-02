# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:09:35 2017

@author: Sarah
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Function to calculate transition rates
def calculate_rate(x, y, temp, state, max_spin):

	#This is to account for periodic boundary conditions
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
K = 0.6
num_spins = 30
num_updates = 50*num_spins*num_spins
#state = 2*np.random.randint(2, size=(num_spins,num_spins))-1

idx = np.linspace(0, num_spins - 1, num_spins)

# plt.imshow(state, cmap='Greys',  interpolation='nearest')
# plt.show()

i = 0

t0 = time.time()


#while(j < num_trials)

# Perform the updates
#while(i < num_updates):
#	row = int(np.random.choice(idx))
#	col = int(np.random.choice(idx))

#	rate = calculate_rate(row, col, K, state, num_spins)
#	draw = np.random.uniform(0, 1)
#if(draw <= rate):
 #                state[row,col] = -state[row,col]
     #    k = 0
      #   s = 0
       #  while(k < num_spins)
        #      if(k == num_spins):
         #         s = s + state[15,k]*state[15,1]
          #        else: 
           #           s = s + state[15,k]*state[15,k+1]
            #    k = i + 1
 #    i = i + 1
 
 
num_trials = 200

g1 = np.zeros([num_trials,num_updates]) 

j = 0
 # Perform the updates
while(j < num_trials):
    state = 2*np.random.randint(2, size=(num_spins,num_spins))-1
    #g1 = np.zeros([num_trials,num_updates]) 
    
    for i in range(num_updates):
        row = int(np.random.choice(idx))
        col = int(np.random.choice(idx))
    
        rate = calculate_rate(row, col, K, state, num_spins)
        draw = np.random.uniform(0, 1)
     
        if(draw <= rate):
             state[row,col] = -state[row,col]
        s = 0     
        for k in range(num_spins-1):
            
            if(k == num_spins-1):
                s = s + state[15,k]*state[15,1]
            else: 
                s = s + state[15,k]*state[15,k+1]
            g1[j,i] = (1/num_spins)*s      
    j = j + 1
    print(j)

rho = (1/2)*(1 - np.mean(g1,axis=0))
length = 1/rho

t1 = time.time()
total = t1 - t0
print('total')

# Show the final configuration
plt.figure(0)
plt.imshow(state, cmap='Greys',  interpolation='nearest')
plt.show()

t = range(num_updates)
model = 0.04*np.power(t,1/2) + 2

plt.figure(1)
plt.plot(t,length)
plt.hold(True)
plt.plot(t,model)




