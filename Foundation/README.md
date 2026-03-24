# Linear Algebra with NumPy — Practice Scripts

A collection of Python scripts for learning and visualizing core linear algebra concepts using NumPy and Matplotlib.

---

## Files

### `vectorbascis.py`
Demonstrates fundamental NumPy array operations:

| Operation | NumPy syntax | What it does |
|-----------|-------------|--------------|
| Vector Addition | `a + b` | Adds element-wise |
| Dot Product | `np.dot(a, b)` | Multiplies element-wise, then sums → scalar |
| Matrix Multiplication | `A @ B` or `np.matmul(A, B)` | Row × column products |
| Transpose | `A.T` or `np.transpose(A)` | Flips rows and columns |

**Run:**
```bash
python vectorbascis.py
```

---

### `spanofvectors.py`
Interactive 3D visualizer for **vector span**, **linear dependence**, and **linear independence**.

**Run:**
```bash
python spanofvectors.py
```

You will be prompted to enter 1–3 vectors in R³ (format: `x y z`).

**What the script outputs:**
- Whether the vectors are **linearly independent or dependent**
- The **rank** of the matrix formed by the vectors
- A geometric description of their **span**
- A **3D interactive plot** showing the vectors as arrows and their span shaded

**Span outcomes by rank:**

| Rank | Span | Visual |
|------|------|--------|
| 0 | Just the origin | Point |
| 1 | Line through origin | Blue line |
| 2 | Plane through origin | Blue shaded plane |
| 3 | All of R³ | Annotated on plot |

**Example inputs to try:**

```
# Two independent vectors → plane
How many vectors? 2
Vector 1: 1 0 0
Vector 2: 0 1 0

# Two dependent vectors → line
How many vectors? 2
Vector 1: 1 2 3
Vector 2: 2 4 6

# Three independent vectors → spans R³
How many vectors? 3
Vector 1: 1 0 0
Vector 2: 0 1 0
Vector 3: 0 0 1
```

---

## Requirements

```bash
pip install numpy matplotlib
```

| Package | Purpose |
|---------|---------|
| `numpy` | Array math, matrix rank, linear algebra |
| `matplotlib` | 3D plotting (`mpl_toolkits.mplot3d`) |

---

## Key Concepts

**Span** — the set of all vectors reachable by scaling and adding a given set of vectors.

**Linear Independence** — vectors are independent if none of them can be written as a combination of the others. Checked via:
```python
rank = np.linalg.matrix_rank(matrix)
is_independent = (rank == number_of_vectors)
```

**Rank** — the number of linearly independent columns (or rows) in a matrix. Tells you the dimension of the span.

**Transpose** — flipping a matrix over its diagonal: element `A[i][j]` moves to `A.T[j][i]`.
