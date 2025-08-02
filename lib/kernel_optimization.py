"""
Kernel Optimization Utilities

This module provides utilities for optimizing CUDA kernel performance,
including tuning of block and grid sizes, memory access patterns, and other
performance-critical parameters.
"""

def calculate_optimal_block_size(data_size, max_threads_per_block=1024):
    """
    Calculate an optimal block size for a given data size.
    
    Args:
        data_size (int): The total size of the data to be processed.
        max_threads_per_block (int): Maximum threads allowed per block (default: 1024).
    
    Returns:
        int: Optimal block size.
    """
    if data_size <= max_threads_per_block:
        return max(1, data_size)
    return max_threads_per_block

def calculate_grid_size(data_size, block_size):
    """
    Calculate the grid size based on data size and block size.
    
    Args:
        data_size (int): The total size of the data to be processed.
        block_size (int): The size of each block.
    
    Returns:
        int: Grid size (number of blocks).
    """
    return (data_size + block_size - 1) // block_size

def suggest_optimization_parameters(data_size, max_threads=1024):
    """
    Suggest optimization parameters for kernel launch.
    
    Args:
        data_size (int): The total size of the data to be processed.
        max_threads (int): Maximum threads per block (default: 1024).
    
    Returns:
        tuple: A tuple of (block_size, grid_size) for kernel launch.
    """
    block_size = calculate_optimal_block_size(data_size, max_threads)
    grid_size = calculate_grid_size(data_size, block_size)
    return block_size, grid_size
