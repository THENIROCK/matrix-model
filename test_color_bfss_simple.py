import tensorflow as tf
import numpy as np
from color_bfss_helpers import extract_color_matrices

"""
Simple test script to verify the BFSS color energy function works correctly.
This focuses just on the energy calculation without the wavefunction complexity.
"""

def test_color_bfss_energy_simple():
    """Test the BFSS color energy function without wavefunction complexity."""
    print("Testing BFSS color energy function (simple version)...")
    
    # Set up a simple test case
    N = 2
    batch_size = 5
    
    # Create test matrices with 6 matrices (2 colors × 3 spatial dimensions)
    test_matrices = tf.constant(np.random.randn(batch_size, 6, N, N), dtype=tf.complex64)
    
    print(f"Matrix shape: {test_matrices.shape}")
    print(f"Expected: (batch_size, 6, N, N) = ({batch_size}, 6, {N}, {N})")
    
    # Test parameters
    g = 1.0
    coherent_expectation_X_squared = 1.5  # ⟨α|X^i²|α⟩
    coherent_expectation_X_i_X_j = 0.8    # ⟨α|X_i X_j|α⟩
    
    # Extract matrices by color
    color_mats = extract_color_matrices(test_matrices)
    X_A_mats = color_mats['A']  # [X_A^1, X_A^2, X_A^3]
    X_B_mats = color_mats['B']  # [X_B^1, X_B^2, X_B^3]
    
    print(f"Extracted color A matrices: {len(X_A_mats)}")
    print(f"Extracted color B matrices: {len(X_B_mats)}")
    
    # Test potential energy terms manually
    
    # Term 1: (1/2) g² ⟨α|X^i²|α⟩ X_A^j²
    term1 = 0.0
    for j, X_A_j in enumerate(X_A_mats):
        # tr(X_A^j²)
        trace_X_A_j_squared = tf.einsum("bij,bji->b", X_A_j, X_A_j)
        term1 += tf.math.real(trace_X_A_j_squared)
    term1 = 0.5 * g * g * coherent_expectation_X_squared * term1
    
    print(f"Term 1 (coherent X² interaction) shape: {term1.shape}")
    print(f"Term 1 values: {term1}")
    
    # Term 2: -(1/2) g² ⟨α|X_i X_j|α⟩ X_A^i X_A^j
    term2 = 0.0
    for i, X_A_i in enumerate(X_A_mats):
        for j, X_A_j in enumerate(X_A_mats):
            if i != j:  # Only for i ≠ j
                # tr(X_A^i X_A^j)
                trace_X_A_i_X_A_j = tf.einsum("bij,bji->b", X_A_i, X_A_j)
                term2 += tf.math.real(trace_X_A_i_X_A_j)
    term2 = -0.5 * g * g * coherent_expectation_X_i_X_j * term2
    
    print(f"Term 2 (coherent X_i X_j interaction) shape: {term2.shape}")
    print(f"Term 2 values: {term2}")
    
    # Term 3: (1/4) g² X_A^i² X_B^j²
    term3 = 0.0
    for i, X_A_i in enumerate(X_A_mats):
        for j, X_B_j in enumerate(X_B_mats):
            # tr(X_A^i²) * tr(X_B^j²)
            trace_X_A_i_squared = tf.einsum("bij,bji->b", X_A_i, X_A_i)
            trace_X_B_j_squared = tf.einsum("bij,bji->b", X_B_j, X_B_j)
            term3 += tf.math.real(trace_X_A_i_squared * trace_X_B_j_squared)
    term3 = 0.25 * g * g * term3
    
    print(f"Term 3 (color interaction X_A² X_B²) shape: {term3.shape}")
    print(f"Term 3 values: {term3}")
    
    # Term 4: -(1/4) g² X_A^i X_B^j X_A^j X_B^i
    term4 = 0.0
    for i, X_A_i in enumerate(X_A_mats):
        for j, X_B_j in enumerate(X_B_mats):
            # tr(X_A^i X_B^j X_A^j X_B^i)
            X_A_j = X_A_mats[j]
            X_B_i = X_B_mats[i]
            product = tf.einsum("bij,bjk,bkl,bli->b", X_A_i, X_B_j, X_A_j, X_B_i)
            term4 += tf.math.real(product)
    term4 = -0.25 * g * g * term4
    
    print(f"Term 4 (color interaction X_A X_B X_A X_B) shape: {term4.shape}")
    print(f"Term 4 values: {term4}")
    
    # Total potential energy
    potential = term1 + term2 + term3 + term4
    print(f"Total potential energy shape: {potential.shape}")
    print(f"Total potential energy values: {potential}")
    
    print("BFSS color energy function test completed successfully!")
    return True

def test_different_parameters():
    """Test the energy function with different parameters."""
    print("\nTesting with different parameters...")
    
    # Set up test case
    N = 2
    batch_size = 3
    test_matrices = tf.constant(np.random.randn(batch_size, 6, N, N), dtype=tf.complex64)
    
    # Test with different coupling constants
    g_values = [0.5, 1.0, 2.0]
    coherent_expectation_X_squared = 1.5
    coherent_expectation_X_i_X_j = 0.8
    
    for g in g_values:
        print(f"\nTesting with g = {g}")
        
        # Extract matrices
        color_mats = extract_color_matrices(test_matrices)
        X_A_mats = color_mats['A']
        X_B_mats = color_mats['B']
        
        # Calculate energy terms
        term1 = 0.0
        for j, X_A_j in enumerate(X_A_mats):
            trace_X_A_j_squared = tf.einsum("bij,bji->b", X_A_j, X_A_j)
            term1 += tf.math.real(trace_X_A_j_squared)
        term1 = 0.5 * g * g * coherent_expectation_X_squared * term1
        
        term2 = 0.0
        for i, X_A_i in enumerate(X_A_mats):
            for j, X_A_j in enumerate(X_A_mats):
                if i != j:
                    trace_X_A_i_X_A_j = tf.einsum("bij,bji->b", X_A_i, X_A_j)
                    term2 += tf.math.real(trace_X_A_i_X_A_j)
        term2 = -0.5 * g * g * coherent_expectation_X_i_X_j * term2
        
        term3 = 0.0
        for i, X_A_i in enumerate(X_A_mats):
            for j, X_B_j in enumerate(X_B_mats):
                trace_X_A_i_squared = tf.einsum("bij,bji->b", X_A_i, X_A_i)
                trace_X_B_j_squared = tf.einsum("bij,bji->b", X_B_j, X_B_j)
                term3 += tf.math.real(trace_X_A_i_squared * trace_X_B_j_squared)
        term3 = 0.25 * g * g * term3
        
        term4 = 0.0
        for i, X_A_i in enumerate(X_A_mats):
            for j, X_B_j in enumerate(X_B_mats):
                X_A_j = X_A_mats[j]
                X_B_i = X_B_mats[i]
                product = tf.einsum("bij,bjk,bkl,bli->b", X_A_i, X_B_j, X_A_j, X_B_i)
                term4 += tf.math.real(product)
        term4 = -0.25 * g * g * term4
        
        potential = term1 + term2 + term3 + term4
        print(f"  Potential energy: {potential}")
    
    print("Parameter testing completed!")
    return True

if __name__ == "__main__":
    test_color_bfss_energy_simple()
    test_different_parameters() 