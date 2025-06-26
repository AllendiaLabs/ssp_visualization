import numpy as np
import matplotlib.pyplot as plt

from sspspace.util import make_good_unitary

def power_ssp(ssp, exponent):
    """
    Raises the SSP to a given power/exponent in the Fourier domain.
    """
    ssp_fft = np.fft.fft(ssp)
    ssp_pow = np.fft.ifft(ssp_fft ** exponent).real
    return ssp_pow

# Parameters
N = 1    # Beware of averaging artifacts!!
D = 200   # Dimensionality of each SSP
exponents = np.linspace(-50, 50, 1000)  # Exponents from -50 to 50

# Generate N unitary SSPs
ssps = np.array([make_good_unitary(D) for _ in range(N)])

# Initialize arrays to store average dot products and cosine similarities
average_dot_products = []
average_cosine_similarities = []

# Precompute the original SSPs raised to the 20th power
ssps_original = np.array([power_ssp(ssp, 20) for ssp in ssps])

# Compute norms of the original SSPs
norms_original = np.linalg.norm(ssps_original, axis=1)

# Loop over exponents
for exponent in exponents:
    # Exponentiate each SSP
    ssps_powered = np.array([power_ssp(ssp, exponent) for ssp in ssps])
    
    # Compute dot products between original and exponentiated SSPs
    dot_products = np.einsum('ij,ij->i', ssps_original, ssps_powered)
    
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
plt.plot(exponents, average_dot_products, label='Average Dot Product')
plt.plot(exponents, average_cosine_similarities, label='Average Cosine Similarity')
plt.title('Average Dot Product and Cosine Similarity between Original and Exponentiated SSPs')
plt.xlabel('Exponent')
plt.ylabel('Value')
plt.legend()
plt.show()
