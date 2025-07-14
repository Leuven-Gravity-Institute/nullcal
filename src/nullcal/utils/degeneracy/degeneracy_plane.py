
# Script to calculate the degeneracy plane basis vectors

import bilby
import numpy as np
import scipy.linalg
from scipy.optimize import minimize_scalar

# Parameters
ra = 0.0
dec = 0.0
psi = 0.0
time = 1e7
polarisation_factor = 0.3j

# Create an Einstein Telescope interferometer
et = bilby.gw.detector.InterferometerList(["ET"])

# Create the antenna pattern vectors
fplus = np.zeros(3)
fcross = np.zeros(3)
for i, interferometer in enumerate(et):
    fplus[i] = interferometer.antenna_response(ra, dec, time, psi, "plus")
    fcross[i] = interferometer.antenna_response(ra, dec, time, psi, "cross")

# Print the results
print("fplus:", fplus)
print("fcross:", fcross)
print("cross: ", np.cross(fplus, fcross))

# Calculate the null space basis vectors
signal_space = fplus + polarisation_factor * fcross
null_space = np.ones(3) / np.sqrt(3)
null_vector = scipy.linalg.null_space(np.stack((signal_space, null_space), axis=0))[:, 0]
print("Null vector:", null_vector)

# print("Check if there exists a basis with real basis vectors:")
# # Project the second null vector onto the plane perpendicular to the null space
# second_basis_vector = null_vector - np.dot(null_space, null_vector) * null_space
# print("Orthogonalised second basis vector:", second_basis_vector)
# # Rotate the second basis vector to ensure that the first element is real
# second_basis_vector *= np.exp(-1j * np.angle(second_basis_vector[0]))
# print("Second basis vector rotated:", second_basis_vector)

# Calculate the basis vectors of the degeneracy plane
true_calibration = np.ones(3) # For simplicity, we assume the true calibration is a unit vector
basis_vectors = np.zeros((6, 4))
basis_vectors[:3, 0] = np.abs(true_calibration) * null_space            # common magnitude calibration is not constrained
basis_vectors[:, 0] /= np.linalg.norm(basis_vectors[:3, 0])             # normalise the first basis vector
basis_vectors[3:, 1] = null_space                                       # common phase calibration is not constrained
basis_vectors[:3, 2] = np.abs(true_calibration) * np.real(null_vector)  # first  unconstrained mode because of polarised signal
basis_vectors[3:, 2] = np.imag(null_vector)                             # first  unconstrained mode because of polarised signal
basis_vectors[:3, 3] = np.abs(true_calibration) * np.imag(null_vector)  # second unconstrained mode because of polarised signal
basis_vectors[3:, 3] = np.real(null_vector)                             # second unconstrained mode because of polarised signal

print("Basis vectors of the degeneracy plane:")
for i in range(4):
    print(f"Vector {i+1}: {basis_vectors[:, i]}")

# Apply gram-schmidt orthogonalisation to the second and third basis vectors (first two are already orthogonal)
for i in range(2, 4):
    for j in range(i):
        basis_vectors[:, i] -= np.dot(basis_vectors[:, j], basis_vectors[:, i]) * basis_vectors[:, j]
    basis_vectors[:, i] /= np.linalg.norm(basis_vectors[:, i])

print("Orthogonalised basis vectors of the degeneracy plane:")
for i in range(4):
    print(f"Orthogonalised Vector {i+1}: {basis_vectors[:, i]}")

# Calculate the constrained directions
constrained_directions = scipy.linalg.null_space(basis_vectors.T)

print("Constrained directions:")
for i in range(constrained_directions.shape[1]):
    print(f"Constrained Direction {i+1}: {constrained_directions[:, i]}")

# Rotate to basis vectors to separate magnitude and phase as much as possible
def rotate_vectors(theta, vectors):
    """
    Rotate the vectors by an angle theta.
    """
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
    return vectors @ rotation_matrix

def cost_function(theta, vectors):
    """
    The first vector is penalised on its phase, the second vector is penalised on its magnitude.
    """
    rotated_vectors = rotate_vectors(theta, vectors)

    return np.sum(rotated_vectors[3:, 0]**2) + np.sum(rotated_vectors[:3, 1]**2)

# Get the optimal rotation angle to separate magnitude and phase for the degeneracy plane
result = minimize_scalar(lambda theta: cost_function(theta, basis_vectors[:, 2:]), bounds=(0, np.pi), method='bounded')
print("Optimal rotation angle (radians):", result.x)

# Apply the optimal rotation to the basis vectors
basis_vectors[:, 2:] = rotate_vectors(result.x, basis_vectors[:, 2:])

print("Basis vectors after optimal rotation:")
for i in range(4):
    print(f"Rotated Vector {i+1}: {basis_vectors[:, i]}")

thetas = np.linspace(0, 2 * np.pi, 1000)
# cost_function_values = [cost_function(theta, basis_vectors[:, 2:]) for theta in thetas]
cost_function_values = [cost_function(theta, constrained_directions) for theta in thetas]
import matplotlib.pyplot as plt
plt.plot(thetas, cost_function_values)
plt.xlabel('Rotation angle (radians)')
plt.ylabel('Cost function value')
plt.title('Cost function vs Rotation angle')
plt.grid()
plt.show()




