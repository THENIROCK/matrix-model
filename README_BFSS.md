# BFSS Matrix Model Implementation

This document explains the minimal adaptations made to the Han and Hartnoll codebase to use the **BFSS (Banks-Fischler-Shenker-Susskind) Hamiltonian** instead of the mini-BMN Hamiltonian.

## What Changed

### 1. Hamiltonian Difference

**Original mini-BMN Hamiltonian:**
```
H = (1/2) tr(ΠᵢΠᵢ) - (1/4) tr([Xᵢ, Xⱼ]²) + (1/2) ν² tr(XᵢXᵢ) + iν εᵢⱼₖ tr(XᵢXⱼXₖ)
```

**New BFSS Hamiltonian:**
```
H = (1/2) tr(ΠᵢΠᵢ) - (1/4) tr([Xᵢ, Xⱼ]²)
```

The BFSS Hamiltonian is simpler - it only has:
- Kinetic term: `(1/2) tr(ΠᵢΠᵢ)`
- Potential term: `-(1/4) tr([Xᵢ, Xⱼ]²)`

No mass deformation or cubic terms.

### 2. Code Changes

#### New Function in `obs.py`
Added `BFSS_bosonic_energy()` function that implements the BFSS Hamiltonian.

#### New Demo Script
Created `demo_bfss.py` - a modified version of `demo.py` that uses the BFSS Hamiltonian.

#### Test Script
Created `test_bfss.py` to verify the implementation works correctly.

## How to Use

### 1. Test the Implementation
```bash
python test_bfss.py
```

### 2. Run BFSS Training
```bash
# Basic usage (N=2 matrices, normalizing flow architecture)
python demo_bfss.py nf 2

# With more options
python demo_bfss.py nf 2 -b 1000 -s 5000 -l 2 -n 2

# Using masked autoregressive flow
python demo_bfss.py maf 2 -b 1000 -s 5000
```

### 3. Command Line Arguments

- `flow`: Architecture type (`nf` for normalizing flow, `maf` for masked autoregressive flow)
- `N`: Size of the matrices (e.g., 2, 4, 6, 8, 10, 12, 14)
- `-b`: Batch size (default: 1000)
- `-s`: Number of training steps (default: 10000)
- `-f`: Number of fermions (default: 0, bosonic only)
- `-r`: Rank of fermion decomposition (default: 1)
- `-l`: Number of hidden layers (default: 1)
- `-n`: Number of mixture components (default: 1)
- `-i`: Path to restore from (optional)
- `--prof`: Enable profiling

### 4. Example Commands

```bash
# Simple BFSS with N=2
python demo_bfss.py nf 2

# BFSS with N=4, more training steps
python demo_bfss.py nf 4 -s 15000

# BFSS with N=2, masked autoregressive flow
python demo_bfss.py maf 2 -s 8000

# BFSS with fermions (if you want to extend to supersymmetric BFSS)
python demo_bfss.py nf 2 -f 2
```

## Key Differences from Original

1. **No mass parameter**: The BFSS model doesn't have a mass parameter `m` like the mini-BMN model
2. **Simpler potential**: Only the commutator squared term, no mass deformation or cubic terms
3. **No fuzzy sphere initialization**: The original code had special initialization near the fuzzy sphere for the BMN model, which is removed for BFSS
4. **Different naming**: Results are saved with "bfss_" prefix to distinguish from BMN results

## Physics Background

The BFSS matrix model is the bosonic part of the BFSS matrix theory, which is a candidate for M-theory in the infinite momentum frame. It describes D0-branes in 11-dimensional M-theory.

The key feature is that it has no mass deformation, so the matrices can spread out more freely compared to the mini-BMN model, which has a confining potential that keeps the matrices near the origin.

## Troubleshooting

1. **Make sure conda environment is activated**: `conda activate py37`
2. **Check TensorFlow installation**: The code requires TensorFlow 1.13
3. **Run tests first**: `python test_bfss.py` to verify everything works
4. **Start with small N**: Try N=2 first before moving to larger matrices

## Files Modified/Created

- **Modified**: `obs.py` (added BFSS energy function)
- **Created**: `demo_bfss.py` (BFSS demo script)
- **Created**: `test_bfss.py` (test script)
- **Created**: `README_BFSS.md` (this file) 