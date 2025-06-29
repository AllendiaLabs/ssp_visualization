import numpy as np
import matplotlib.pyplot as plt

from utils import make_good_unitary, power_ssp

# Parameters
N = 1    # Beware of averaging artifacts!!
D = 30   # Dimensionality of each SSP
exponents = np.linspace(-50, 50, 1000)  # Exponents from -50 to 50

# Generate N unitary SSPs
ssps = np.array([make_good_unitary(D) for _ in range(N)])

# Initialize arrays to store average dot products and cosine similarities
average_dot_products = []
average_cosine_similarities = []

# Compute norms of the original SSPs
norms_original = np.linalg.norm(ssps, axis=1)

# Loop over exponents
for exponent in exponents:
    # Exponentiate each SSP
    ssps_powered = np.array([power_ssp(ssp, exponent) for ssp in ssps])
    
    # Compute dot products between original and exponentiated SSPs
    dot_products = np.einsum('ij,ij->i', ssps, ssps_powered)
    
    # Compute norms of the exponentiated SSPs
    norms_powered = np.linalg.norm(ssps_powered, axis=1)
    
    # Compute cosine similarities
    cosine_similarities = dot_products / (norms_original * norms_powered)
    
    # Compute average dot product and average cosine similarity
    average_dot_product = np.mean(dot_products)
    average_cosine_similarity = np.mean(cosine_similarities)
    
    average_dot_products.append(average_dot_product)
    average_cosine_similarities.append(average_cosine_similarity)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(exponents, average_dot_products, label='Dot Product')
plt.plot(exponents, average_cosine_similarities, label='Cosine Similarity')
plt.title('Dot Product and Cosine Similarity between Original and Exponentiated SSPs')
plt.xlabel('Exponent')
plt.ylabel('Value')
plt.legend()
plt.show()
