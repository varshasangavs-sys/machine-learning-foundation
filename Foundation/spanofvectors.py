"""
spanofvectors.py
================
Interactive 3D visualizer for vector span, linear dependence/independence.

Usage:
  python spanofvectors.py

You will be prompted to enter 1–3 vectors in R³.
The script will:
  1. Print whether the vectors are linearly dependent or independent.
  2. Describe their span (point / line / plane / all of R³).
  3. Show a 3D plot with:
       - The vectors drawn as arrows from the origin
       - The span shaded (line for rank-1, plane for rank-2)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---------------------------------------------------------------------------
# 1.  INPUT
# ---------------------------------------------------------------------------

def get_vectors():
    """Prompt the user for 1–3 vectors in R³ and return them as a list of
    numpy arrays."""
    while True:
        try:
            n = int(input("How many vectors do you want to enter? (1 / 2 / 3): ").strip())
            if n in (1, 2, 3):
                break
            print("  Please enter 1, 2, or 3.")
        except ValueError:
            print("  Invalid input — enter a whole number.")

    vectors = []
    for i in range(n):
        while True:
            raw = input(f"  Vector {i + 1}  (enter x y z separated by spaces): ").strip()
            parts = raw.split()
            if len(parts) == 3:
                try:
                    v = np.array([float(p) for p in parts])
                    vectors.append(v)
                    break
                except ValueError:
                    pass
            print("  Please enter exactly 3 numbers, e.g.  1 0 0")
    return vectors


# ---------------------------------------------------------------------------
# 2.  ANALYSIS
# ---------------------------------------------------------------------------

def analyse(vectors):
    """
    Determine rank, linear dependence, and a human-readable span description.

    Returns
    -------
    rank            : int
    is_independent  : bool
    span_label      : str   (short label used in the plot title)
    explanation     : list of str  (detailed console lines)
    """
    n = len(vectors)
    M = np.column_stack(vectors)          # each vector is a column
    rank = int(np.linalg.matrix_rank(M))

    is_independent = (rank == n)

    # --- span label ---
    span_map = {
        0: "just the origin {0}",
        1: "a Line through the origin",
        2: "a Plane through the origin",
        3: "all of R³",
    }
    span_label = span_map[rank]

    # --- detailed explanation lines ---
    lines = []
    lines.append("")
    lines.append("=" * 55)
    lines.append("  ANALYSIS")
    lines.append("=" * 55)
    for i, v in enumerate(vectors):
        lines.append(f"  v{i + 1} = {v}")
    lines.append(f"\n  Matrix of vectors (columns):\n{M}")
    lines.append(f"\n  Rank = {rank}   (out of {n} vectors)")
    lines.append("")

    if is_independent:
        lines.append("  LINEARLY INDEPENDENT")
        lines.append("  Reason: rank equals the number of vectors.")
        lines.append("  No vector can be written as a combination of the others.")
    else:
        lines.append("  LINEARLY DEPENDENT")
        lines.append(f"  Reason: rank ({rank}) < number of vectors ({n}).")
        lines.append("  At least one vector is a linear combination of the others.")

    lines.append(f"\n  Span  =>  {span_label}")

    # explain what the span means geometrically
    if rank == 0:
        lines.append("  All vectors are zero — span is just the origin.")
    elif rank == 1:
        lines.append("  All vectors lie on the same line through the origin.")
        lines.append("  Span = { t·v₁ | t ∈ R }")
    elif rank == 2:
        lines.append("  Vectors span a 2-D plane through the origin.")
        lines.append("  Span = { s·v₁ + t·v₂ | s, t ∈ R }  (using two independent ones)")
    else:
        lines.append("  Vectors span all of 3-D space.")
        lines.append("  Every point in R³ can be reached by a linear combination.")

    lines.append("=" * 55)
    return rank, is_independent, span_label, lines


# ---------------------------------------------------------------------------
# 3.  PLOT HELPERS
# ---------------------------------------------------------------------------

COLORS = ["#e63946", "#2a9d8f", "#e9c46a"]   # red, teal, yellow


def _axis_limit(vectors):
    """Return a nice axis limit based on the longest vector."""
    max_norm = max(np.linalg.norm(v) for v in vectors)
    return max(2.0, max_norm * 1.5)


def _draw_vectors(ax, vectors):
    """Draw each vector as a thick arrow from the origin."""
    lim = _axis_limit(vectors)
    for i, v in enumerate(vectors):
        ax.quiver(0, 0, 0, v[0], v[1], v[2],
                  color=COLORS[i % len(COLORS)],
                  arrow_length_ratio=0.15,
                  linewidth=2.5,
                  label=f"v{i + 1} = {v}")
        # label tip
        ax.text(v[0] * 1.05, v[1] * 1.05, v[2] * 1.05,
                f"v{i + 1}", color=COLORS[i % len(COLORS)], fontsize=11, fontweight='bold')


def _draw_span_rank1(ax, vectors, lim):
    """Draw the line spanned by the first non-zero vector."""
    # find first non-zero vector
    base = None
    for v in vectors:
        if np.linalg.norm(v) > 1e-10:
            base = v / np.linalg.norm(v)
            break
    if base is None:
        return
    t = np.linspace(-lim, lim, 200)
    pts = np.outer(t, base)
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
            color="royalblue", linewidth=2, alpha=0.6, label="Span (line)")


def _draw_span_rank2(ax, vectors, lim):
    """Draw the plane spanned by two independent vectors."""
    # pick two linearly independent vectors
    basis = []
    for v in vectors:
        if len(basis) == 0:
            if np.linalg.norm(v) > 1e-10:
                basis.append(v)
        elif len(basis) == 1:
            # check independence
            cross = np.cross(basis[0], v)
            if np.linalg.norm(cross) > 1e-10:
                basis.append(v)
                break

    if len(basis) < 2:
        return  # fallback — shouldn't happen

    b1 = basis[0] / np.linalg.norm(basis[0])
    # orthogonalise b2 w.r.t. b1 for a nicer grid
    b2 = basis[1] - np.dot(basis[1], b1) * b1
    if np.linalg.norm(b2) > 1e-10:
        b2 /= np.linalg.norm(b2)
    else:
        b2 = basis[1]

    s = np.linspace(-lim, lim, 12)
    t = np.linspace(-lim, lim, 12)
    S, T = np.meshgrid(s, t)

    X = S * b1[0] + T * b2[0]
    Y = S * b1[1] + T * b2[1]
    Z = S * b1[2] + T * b2[2]

    ax.plot_surface(X, Y, Z, alpha=0.25, color="royalblue",
                    label="Span (plane)")
    # wireframe on top for clarity
    ax.plot_wireframe(X, Y, Z, alpha=0.15, color="royalblue", linewidth=0.5)


def _draw_span_rank3(ax, lim):
    """Annotate that the span is all of R³ — we cannot shade infinite space."""
    ax.text2D(0.5, 0.01, "Span = all of R³  (full 3-D space)",
              transform=ax.transAxes, ha="center", fontsize=10,
              color="royalblue", style="italic")


# ---------------------------------------------------------------------------
# 4.  MAIN PLOT
# ---------------------------------------------------------------------------

def plot_3d(vectors, rank, is_independent, span_label):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    lim = _axis_limit(vectors)

    # draw origin
    ax.scatter(0, 0, 0, color='black', s=40, zorder=5)
    ax.text(0, 0, 0, "  O", fontsize=9, color='black')

    # draw span first (behind vectors)
    if rank == 1:
        _draw_span_rank1(ax, vectors, lim)
    elif rank == 2:
        _draw_span_rank2(ax, vectors, lim)
    elif rank == 3:
        _draw_span_rank3(ax, lim)

    # draw vectors
    _draw_vectors(ax, vectors)

    # axes appearance
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("X", labelpad=8)
    ax.set_ylabel("Y", labelpad=8)
    ax.set_zlabel("Z", labelpad=8)

    dep_str = "Linearly INDEPENDENT" if is_independent else "Linearly DEPENDENT"
    ax.set_title(f"{dep_str}\nSpan  →  {span_label}", fontsize=12, fontweight='bold', pad=14)

    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5.  ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 55)
    print("  3D Vector Span Visualizer")
    print("  Explore linear dependence and the span of vectors")
    print("=" * 55)

    vectors = get_vectors()
    rank, is_independent, span_label, explanation = analyse(vectors)

    for line in explanation:
        print(line)

    print("\n  Opening 3D plot …  (close the window to exit)\n")
    plot_3d(vectors, rank, is_independent, span_label)


if __name__ == "__main__":
    main()
