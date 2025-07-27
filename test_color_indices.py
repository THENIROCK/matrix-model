import tensorflow as tf
import numpy as np

"""
Test script to understand how to handle color indices in the new BFSS Hamiltonian.
"""

def test_color_index_structure():
    """Test the color index structure for the new BFSS Hamiltonian."""
    print("Testing color index structure...")
    
    # Set up test parameters
    batch_size = 2
    N = 3  # matrix size
    num_colors = 2  # A, B = 1, 2
    num_spatial = 3  # i, j = 1, 2, 3
    total_matrices = num_colors * num_spatial  # 6 matrices total
    
    # Create test matrices with the new structure
    # Shape: (batch_size, 6, N, N) where 6 = 2 colors × 3 spatial dimensions
    test_matrices = tf.constant(np.random.randn(batch_size, total_matrices, N, N), dtype=tf.complex64)
    
    print(f"Matrix shape: {test_matrices.shape}")
    print(f"Expected: (batch_size, {total_matrices}, N, N) = ({batch_size}, {total_matrices}, {N}, {N})")
    
    # Extract matrices by color and spatial index
    # Color A (index 0) matrices
    X_A_1 = test_matrices[:, 0, :, :]  # X_A^1
    X_A_2 = test_matrices[:, 1, :, :]  # X_A^2  
    X_A_3 = test_matrices[:, 2, :, :]  # X_A^3
    
    # Color B (index 1) matrices
    X_B_1 = test_matrices[:, 3, :, :]  # X_B^1
    X_B_2 = test_matrices[:, 4, :, :]  # X_B^2
    X_B_3 = test_matrices[:, 5, :, :]  # X_B^3
    
    print(f"\nExtracted matrices:")
    print(f"X_A_1 shape: {X_A_1.shape}")  # Should be (batch_size, N, N)
    print(f"X_A_2 shape: {X_A_2.shape}")
    print(f"X_A_3 shape: {X_A_3.shape}")
    print(f"X_B_1 shape: {X_B_1.shape}")
    print(f"X_B_2 shape: {X_B_2.shape}")
    print(f"X_B_3 shape: {X_B_3.shape}")
    
    # Test some basic operations
    print(f"\nTesting basic operations:")
    
    # Sum over spatial indices for each color
    sum_A = X_A_1 + X_A_2 + X_A_3
    sum_B = X_B_1 + X_B_2 + X_B_3
    print(f"Sum over spatial indices for color A: {sum_A.shape}")
    print(f"Sum over spatial indices for color B: {sum_B.shape}")
    
    # Sum over colors for each spatial index
    sum_1 = X_A_1 + X_B_1  # sum over colors for spatial index 1
    sum_2 = X_A_2 + X_B_2  # sum over colors for spatial index 2
    sum_3 = X_A_3 + X_B_3  # sum over colors for spatial index 3
    print(f"Sum over colors for spatial index 1: {sum_1.shape}")
    print(f"Sum over colors for spatial index 2: {sum_2.shape}")
    print(f"Sum over colors for spatial index 3: {sum_3.shape}")
    
    # Test matrix multiplication between different colors
    product_AB_1 = tf.einsum("bij,bjk->bik", X_A_1, X_B_1)  # X_A^1 * X_B^1
    print(f"Product X_A^1 * X_B^1: {product_AB_1.shape}")
    
    # Test commutator between different colors
    commutator_AB_1 = tf.einsum("bij,bjk->bik", X_A_1, X_B_1) - tf.einsum("bij,bjk->bik", X_B_1, X_A_1)
    print(f"Commutator [X_A^1, X_B^1]: {commutator_AB_1.shape}")
    
    print("\nColor index structure test completed successfully!")
    return True

def test_coherent_state_expectations():
    """Test how to handle coherent state expectation values."""
    print("\nTesting coherent state expectation values...")
    
    # For now, let's assume the coherent state expectation values are constants
    # In the real implementation, these would be computed from the coherent state |α⟩
    
    # Example coherent state expectation values (these would be computed from |α⟩)
    coherent_expectation_X_squared = 1.5  # ⟨α|X^i²|α⟩ (same for all i)
    coherent_expectation_X_i_X_j = 0.8    # ⟨α|X_i X_j|α⟩ (same for all i,j)
    
    print(f"Coherent state expectation values:")
    print(f"⟨α|X^i²|α⟩ = {coherent_expectation_X_squared}")
    print(f"⟨α|X_i X_j|α⟩ = {coherent_expectation_X_i_X_j}")
    
    # These would be used as constants in the Hamiltonian
    print("These would be used as constants in the new BFSS Hamiltonian.")
    
    print("Coherent state expectation test completed!")
    return True

if __name__ == "__main__":
    test_color_index_structure()
    test_coherent_state_expectations() 