# Machine Learning — Foundation

Scripts and notebooks for learning core ML and linear algebra concepts from scratch.

---

## Projects

### `spam_email_classifier/`

A Naive Bayes spam classifier built from scratch in a Jupyter notebook — no sklearn, just math and Python.

**Files:**
- `naivbayers_classifer.ipynb` — the full classifier notebook
- `spam.csv` — dataset of 5,572 labelled SMS messages (spam / ham)

**What it does:**
1. Loads and explores the dataset (~13% spam, ~87% ham)
2. Calculates prior probabilities $P(\text{spam})$ and $P(\text{ham})$
3. Preprocesses messages — lowercase, strip punctuation, remove stop words
4. Builds word frequency dictionaries for spam and ham
5. Applies Laplace (add-1) smoothing to handle unseen words
6. Predicts using log-probabilities to avoid numerical underflow
7. Evaluates with accuracy, precision, recall and F1 score

**Run:**
```bash
jupyter notebook spam_email_classifier/naivbayers_classifer.ipynb
```

**Requirements:**
```bash
pip install pandas
```

---

## Scripts

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
Interactive 3D visualizer for vector span, linear dependence, and linear independence.

**Run:**
```bash
python spanofvectors.py
```

You'll be prompted to enter 1–3 vectors in R³ (format: `x y z`).

**What it outputs:**
- Whether the vectors are linearly independent or dependent
- The rank of the matrix formed by the vectors
- A geometric description of their span
- A 3D interactive plot showing the vectors as arrows and their span shaded

| Rank | Span | Visual |
|------|------|--------|
| 0 | Just the origin | Point |
| 1 | Line through origin | Blue line |
| 2 | Plane through origin | Blue shaded plane |
| 3 | All of R³ | Annotated on plot |

**Example inputs:**
```
# Two independent vectors → plane
How many vectors? 2
Vector 1: 1 0 0
Vector 2: 0 1 0

# Two dependent vectors → line
How many vectors? 2
Vector 1: 1 2 3
Vector 2: 2 4 6
```

**Requirements:**
```bash
pip install numpy matplotlib
```

---

## Key Concepts Covered

**Naive Bayes** — probabilistic classifier using Bayes' theorem. Assumes word independence and computes the most likely class given the observed words.

**Laplace Smoothing** — adding 1 to word counts so unseen words don't zero out the probability.

**Log Probabilities** — using log-sum instead of product to avoid floating point underflow.

**Span** — the set of all vectors reachable by scaling and adding a given set of vectors.

**Linear Independence** — checked via matrix rank:
```python
rank = np.linalg.matrix_rank(matrix)
is_independent = (rank == number_of_vectors)
```

**Rank** — the number of linearly independent columns in a matrix. Determines the dimension of the span.
