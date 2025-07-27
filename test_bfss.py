import tensorflow as tf
import numpy as np
from obs import *
from wavefunc import *
from algebra import *
from dist import *
import tensorflow_probability as tfp

"""
Simple test script to verify the BFSS implementation works correctly.
"""

def test_bfss_energy():
    """Test that the BFSS energy function works correctly."""
    print("Testing BFSS energy function...")
    
    # Set up a simple test case
    N = 2
    batch_size = 10
    algebra = SU(N)
    
    # Create some test matrices
    test_matrices = tf.constant(np.random.randn(batch_size, 3, N, N), dtype=tf.complex64)
    
    # Test the basic matrix functions
    quadratic = matrix_quadratic_potential(test_matrices)
    commutator = matrix_commutator_square(test_matrices)
    
    print(f"Matrix shape: {test_matrices.shape}")
    print(f"Quadratic potential shape: {quadratic.shape}")
    print(f"Commutator squared shape: {commutator.shape}")
    
    # Test with a simple wavefunction
    with tf.variable_scope("test_bfss"):
        bosonic_dim = 2 * algebra.dim
        vectorizer = Vectorizer(algebra, tfp.bijectors.Softplus())
        bosonic_wavefunc = NormalizingFlow([Normal()] * bosonic_dim, 0, tfp.bijectors.Sigmoid())
        fermionic_wavefunc = FermionicWavefunction(algebra, bosonic_dim, 2, 0, 1, 10, 1)
        wavefunc = Wavefunction(algebra, vectorizer, bosonic_wavefunc, fermionic_wavefunc)
        
        # Test BFSS energy
        energy = BFSS_bosonic_energy(wavefunc, test_matrices)
        print(f"BFSS energy shape: {energy.shape}")
        
        # Test potential only
        energy_potential = BFSS_bosonic_energy(wavefunc, test_matrices, potential_only=True)
        print(f"BFSS potential-only energy shape: {energy_potential.shape}")
    
    print("BFSS energy function test completed successfully!")
    return True

if __name__ == "__main__":
    test_bfss_energy() 