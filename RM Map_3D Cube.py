# code to simulate the electron density n_e and the B to calculate RM map for a sample cube

import numpy as np
import matplotlib.pyplot as plt
# creating a hypothetical 3d cube of a B field
size = 100
b_field_z = np.random.normal(loc=0.5, scale=1.0, size=(size, size, size))
# simulating electron density (n_e)
x, y, z = np.indices((size, size, size))
center = size // 2
radius = np.sqrt((x-center)**2 + (y-center)**2 + (z-center)**2)
n_e = np.exp(-radius/20) # exponential drop-off from center
# collapse the line of sight integration
# RM (strength and direction of B_field) is defined as the (constant) times the, 
# (the sum over the distance of the source of the electron density and the Bfield that's parallel to the direction of propagation)
# use numerical method of integration: np.trapz or np.sum
rotation_measure_map = np.sum(n_e * b_field_z, axis=2)
# visualizing it
plt. imshow(rotation_measure_map, cmap='RdBu')
plt.colorbar(label= 'Rotation Measure (RM)')
plt.title("Simulated RM Map")
plt.show()
