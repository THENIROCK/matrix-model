import tensorflow as tf
import numpy as np

"""
Helper functions for the new BFSS Hamiltonian with color indices.
This handles the structure where we have 6 matrices: 2 colors × 3 spatial dimensions.
"""

def extract_color_matrices(mats):
    """
    Extract matrices by color and spatial index from the 6-matrix structure.
    
    Arguments:
        mats (tensor of shape (batch_size, 6, N, N)): matrices with color and spatial indices
        
    Returns:
        dict: Dictionary containing extracted matrices
            - 'A': color A matrices (X_A^1, X_A^2, X_A^3)
            - 'B': color B matrices (X_B^1, X_B^2, X_B^3)
    """
    assert mats.shape[1] == 6, f"Expected 6 matrices, got {mats.shape[1]}"
    
    # Color A matrices (indices 0, 1, 2)
    X_A_1 = mats[:, 0, :, :]  # X_A^1
    X_A_2 = mats[:, 1, :, :]  # X_A^2  
    X_A_3 = mats[:, 2, :, :]  # X_A^3
    
    # Color B matrices (indices 3, 4, 5)
    X_B_1 = mats[:, 3, :, :]  # X_B^1
    X_B_2 = mats[:, 4, :, :]  # X_B^2
    X_B_3 = mats[:, 5, :, :]  # X_B^3
    
    return {
        'A': [X_A_1, X_A_2, X_A_3],
        'B': [X_B_1, X_B_2, X_B_3]
    }

def matrix_quadratic_potential_color(mats):
    """
    Compute sum_A sum_i tr(X_A^i)^2 for the color-indexed matrices.
    
    Arguments:
        mats (tensor of shape (batch_size, 6, N, N)): matrices with color and spatial indices
        
    Returns:
        tensor of shape (batch_size,): quadratic potential for each sample
    """
    # Extract matrices by color
    color_mats = extract_color_matrices(mats)
    
    # Sum over colors and spatial indices
    total = 0.0
    for color in ['A', 'B']:
        for i, mat in enumerate(color_mats[color]):
            # tr(X_A^i)^2 or tr(X_B^i)^2
            trace_squared = tf.einsum("bij,bji->b", mat, mat)
            total += tf.math.real(trace_squared)
    
    return total

def matrix_commutator_square_color(mats):
    """
    Compute sum_A,B sum_i,j tr([X_A^i, X_B^j]^2) for the color-indexed matrices.
    
    Arguments:
        mats (tensor of shape (batch_size, 6, N, N)): matrices with color and spatial indices
        
    Returns:
        tensor of shape (batch_size,): commutator squared for each sample
    """
    # Extract matrices by color
    color_mats = extract_color_matrices(mats)
    
    total = 0.0
    # Sum over all color pairs and spatial indices
    for color_A in ['A', 'B']:
        for color_B in ['A', 'B']:
            for i, mat_A in enumerate(color_mats[color_A]):
                for j, mat_B in enumerate(color_mats[color_B]):
                    # Compute commutator [X_A^i, X_B^j]
                    commutator = tf.einsum("bij,bjk->bik", mat_A, mat_B) - tf.einsum("bij,bjk->bik", mat_B, mat_A)
                    # Compute tr([X_A^i, X_B^j]^2)
                    commutator_squared = tf.einsum("bij,bji->b", commutator, commutator)
                    total += tf.math.real(commutator_squared)
    
    return total

def test_color_helpers():
    """Test the helper functions."""
    print("Testing color helper functions...")
    
    # Create test matrices
    batch_size = 2
    N = 3
    test_matrices = tf.constant(np.random.randn(batch_size, 6, N, N), dtype=tf.complex64)
    
    print(f"Test matrices shape: {test_matrices.shape}")
    
    # Test extraction
    color_mats = extract_color_matrices(test_matrices)
    print(f"Extracted color A matrices: {len(color_mats['A'])}")
    print(f"Extracted color B matrices: {len(color_mats['B'])}")
    print(f"Color A matrix 1 shape: {color_mats['A'][0].shape}")
    
    # Test quadratic potential
    quad_pot = matrix_quadratic_potential_color(test_matrices)
    print(f"Quadratic potential shape: {quad_pot.shape}")
    print(f"Quadratic potential values: {quad_pot}")
    
    # Test commutator squared
    comm_squared = matrix_commutator_square_color(test_matrices)
    print(f"Commutator squared shape: {comm_squared.shape}")
    print(f"Commutator squared values: {comm_squared}")
    
    print("Color helper functions test completed!")
    return True

if __name__ == "__main__":
    test_color_helpers() 