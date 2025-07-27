import tensorflow as tf
import numpy as np
from obs import *
from wavefunc import *
from algebra import *
from dist import *
import tensorflow_probability as tfp

"""
Test script to verify the BFSS color energy function works correctly.
"""

def test_color_bfss_energy():
    """Test that the BFSS color energy function works correctly."""
    print("Testing BFSS color energy function...")
    
    # Set up a simple test case
    N = 2
    batch_size = 10
    algebra = SU(N)
    
    # Create test matrices with 6 matrices (2 colors × 3 spatial dimensions)
    test_matrices = tf.constant(np.random.randn(batch_size, 6, N, N), dtype=tf.complex64)
    
    print(f"Matrix shape: {test_matrices.shape}")
    print(f"Expected: (batch_size, 6, N, N) = ({batch_size}, 6, {N}, {N})")
    
    # Test with a simple wavefunction
    with tf.variable_scope("test_color_bfss"):
        bosonic_dim = 2 * algebra.dim * 6  # 6 matrices instead of 3
        vectorizer = Vectorizer(algebra, tfp.bijectors.Softplus())
        bosonic_wavefunc = NormalizingFlow([Normal()] * bosonic_dim, 0, tfp.bijectors.Sigmoid())
        fermionic_wavefunc = FermionicWavefunction(algebra, bosonic_dim, 2, 0, 1, 10, 1)
        wavefunc = Wavefunction(algebra, vectorizer, bosonic_wavefunc, fermionic_wavefunc)
        
        # Test parameters
        g = 1.0
        coherent_expectation_X_squared = 1.5  # ⟨α|X^i²|α⟩
        coherent_expectation_X_i_X_j = 0.8    # ⟨α|X_i X_j|α⟩
        
        # Test BFSS color energy
        energy = BFSS_color_energy(g, coherent_expectation_X_squared, coherent_expectation_X_i_X_j, 
                                  wavefunc, test_matrices)
        print(f"BFSS color energy shape: {energy.shape}")
        
        # Test potential only
        energy_potential = BFSS_color_energy(g, coherent_expectation_X_squared, coherent_expectation_X_i_X_j, 
                                            wavefunc, test_matrices, potential_only=True)
        print(f"BFSS color potential-only energy shape: {energy_potential.shape}")
        
        # Test with different parameters
        g2 = 2.0
        energy2 = BFSS_color_energy(g2, coherent_expectation_X_squared, coherent_expectation_X_i_X_j, 
                                   wavefunc, test_matrices)
        print(f"BFSS color energy with g={g2} shape: {energy2.shape}")
    
    print("BFSS color energy function test completed successfully!")
    return True

def test_color_structure():
    """Test the color index structure more thoroughly."""
    print("\nTesting color index structure...")
    
    # Create test matrices
    batch_size = 2
    N = 2
    test_matrices = tf.constant(np.random.randn(batch_size, 6, N, N), dtype=tf.complex64)
    
    # Import color helpers
    from color_bfss_helpers import extract_color_matrices
    
    # Extract matrices
    color_mats = extract_color_matrices(test_matrices)
    X_A_mats = color_mats['A']  # [X_A^1, X_A^2, X_A^3]
    X_B_mats = color_mats['B']  # [X_B^1, X_B^2, X_B^3]
    
    print(f"Color A matrices: {len(X_A_mats)}")
    print(f"Color B matrices: {len(X_B_mats)}")
    print(f"Color A matrix 1 shape: {X_A_mats[0].shape}")
    print(f"Color B matrix 1 shape: {X_B_mats[0].shape}")
    
    # Test some basic operations
    # Sum over spatial indices for each color
    sum_A = X_A_mats[0] + X_A_mats[1] + X_A_mats[2]
    sum_B = X_B_mats[0] + X_B_mats[1] + X_B_mats[2]
    print(f"Sum over spatial indices for color A: {sum_A.shape}")
    print(f"Sum over spatial indices for color B: {sum_B.shape}")
    
    # Test matrix multiplication between different colors
    product_AB = tf.einsum("bij,bjk->bik", X_A_mats[0], X_B_mats[0])
    print(f"Product X_A^1 * X_B^1: {product_AB.shape}")
    
    print("Color structure test completed!")
    return True

if __name__ == "__main__":
    test_color_structure()
    test_color_bfss_energy() 