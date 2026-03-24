import numpy as np

# =============================================================================
# VECTOR ADDITION
#Vectors are represented in form of  arrays 
# Adding two vectors element-wise. Each element at position i in vector 'a'
# is added to the corresponding element at position i in vector 'b'.
# Example: a=[1,2,3], b=[4,5,6] => a+b = [1+4, 2+5, 3+6] = [5, 7, 9]
# =============================================================================
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result_add = a + b  # NumPy overloads '+' to perform element-wise addition

print("=" * 40)
print("VECTOR ADDITION")
print("=" * 40)
print(f"  a       = {a}")
print(f"  b       = {b}")
print(f"  a + b   = {result_add}")
print()

# =============================================================================
# DOT PRODUCT
# The dot product multiplies corresponding elements and sums all the results.
# Formula: a · b = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
# Example: a=[1,2,3], b=[4,5,6]
#          => (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
# It measures how much two vectors point in the same direction.
# Result: a scalar (single number), not a vector.
# =============================================================================
result_dot = np.dot(a, b)  # Computes the inner product of two arrays

print("=" * 40)
print("DOT PRODUCT")
print("=" * 40)
print(f"  a       = {a}")
print(f"  b       = {b}")
print(f"  a · b   = {a[0]}×{b[0]} + {a[1]}×{b[1]} + {a[2]}×{b[2]} = {result_dot}")
print()

# =============================================================================
# MATRIX MULTIPLICATION
# Each element [i][j] of the result is computed as the dot product of
# row i from matrix A and column j from matrix B.
#
# A (2×2) @ B (2×2) => C (2×2)
#
# Example:
#   A = [[1, 2],    B = [[5, 6],
#        [3, 4]]         [7, 8]]
#
#   C[0][0] = 1×5 + 2×7 = 5 + 14 = 19
#   C[0][1] = 1×6 + 2×8 = 6 + 16 = 22
#   C[1][0] = 3×5 + 4×7 = 15 + 28 = 43
#   C[1][1] = 3×6 + 4×8 = 18 + 32 = 50
#
# Note: '@' is the matrix multiplication operator (same as np.matmul).
#       It is NOT element-wise — order matters: A @ B ≠ B @ A in general.
# =============================================================================
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# result_matmul = A @ B 
 # Equivalent to np.matmul(A, B)
result_matmul=np.matmul(A,B)

print("=" * 40)
print("MATRIX MULTIPLICATION")
print("=" * 40)
print(f"  A =\n{A}")
print(f"\n  B =\n{B}")
print(f"\n  A @ B =\n{result_matmul}")
print()
print("  Verification of each element:")
print(f"    C[0][0] = {A[0,0]}×{B[0,0]} + {A[0,1]}×{B[1,0]} = {A[0,0]*B[0,0]} + {A[0,1]*B[1,0]} = {result_matmul[0,0]}")
print(f"    C[0][1] = {A[0,0]}×{B[0,1]} + {A[0,1]}×{B[1,1]} = {A[0,0]*B[0,1]} + {A[0,1]*B[1,1]} = {result_matmul[0,1]}")
print(f"    C[1][0] = {A[1,0]}×{B[0,0]} + {A[1,1]}×{B[1,0]} = {A[1,0]*B[0,0]} + {A[1,1]*B[1,0]} = {result_matmul[1,0]}")
print(f"    C[1][1] = {A[1,0]}×{B[0,1]} + {A[1,1]}×{B[1,1]} = {A[1,0]*B[0,1]} + {A[1,1]*B[1,1]} = {result_matmul[1,1]}")



# =============================================================================
# TRANSPOSE
# Transposing a matrix flips it over its diagonal — rows become columns
# and columns become rows.
# For a matrix of shape (m × n), the transpose has shape (n × m).
#
# Example (3×3):
#   A   = [[2, 6, 8],      A.T = [[2, 5, 5],
#          [5, 7, 0],              [6, 7, 7],
#          [5, 7, 8]]              [8, 0, 8]]
#
# Element rule: A.T[i][j] = A[j][i]  (row/col indices are swapped)
# Two ways in NumPy:
#   - A.T            (attribute shorthand)
#   - np.transpose(A)  (explicit function, identical result)
# =============================================================================
A = np.array([[2, 6, 8],
              [5, 7, 0],
              [5, 7, 8]])

A_transpose = A.T  # same as np.transpose(A)

print("=" * 40)
print("TRANSPOSE")
print("=" * 40)
print(f"  A =\n{A}")
print(f"\n  A.T =\n{A_transpose}")
print()
print("  Element rule: A.T[i][j] = A[j][i]")
print(f"  A[0][1] = {A[0,1]}  =>  A.T[1][0] = {A_transpose[1,0]}")
print(f"  A[1][2] = {A[1,2]}  =>  A.T[2][1] = {A_transpose[2,1]}")
