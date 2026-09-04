#!/usr/bin/env python3
"""
Plot linear, superlinear, and quadratic convergence on a semilog-y plot.

- Linear:        e_{k+1} = ρ * e_k,          0 < ρ < 1
- Superlinear:   e_{k+1} = e_k^p,            1 < p < 2   (e.g., p = 1.5)
- Quadratic:     e_{k+1} = e_k^2

These model typical rates seen in practice: e.g., gradient descent with
strong convexity exhibits linear convergence, quasi-Newton methods are
often superlinear, and Newton's method is locally quadratic.
"""
import numpy as np
import matplotlib.pyplot as plt


def make_sequence(kind, kmax, e0=1e-1, rho=0.7, p=1.5):
    """Return an array [e_0, ..., e_kmax] following the chosen rate."""
    e = np.empty(kmax + 1, dtype=float)
    e[0] = e0
    for k in range(kmax):
        if kind == "linear":
            e[k + 1] = rho * e[k]
        elif kind == "superlinear":
            # order-p (1 < p < 2) superlinear decay
            e[k + 1] = e[k] ** p
        elif kind == "quadratic":
            e[k + 1] = e[k] ** 2
        else:
            raise ValueError("kind must be 'linear', 'superlinear', or 'quadratic'")
    return e


def main():
    kmax = 8           # keep modest so quadratic doesn't underflow to 0
    e0 = 1e-1          # initial error
    rho = 0.7          # linear convergence ratio (0 < rho < 1)
    p = 1.5            # superlinear order (1 < p < 2)

    k = np.arange(kmax + 1)

    e_lin  = make_sequence("linear",      kmax, e0=e0, rho=rho)
    e_sup  = make_sequence("superlinear", kmax, e0=e0, p=p)
    e_quad = make_sequence("quadratic",   kmax, e0=e0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(k, e_lin,  "-o", label=f"Linear (ρ={rho})",    linewidth=2, markersize=5)
    ax.semilogy(k, e_sup,  "-s", label=f"Superlinear (p={p})", linewidth=2, markersize=5)
    ax.semilogy(k, e_quad, "-^", label="Quadratic (p=2)",      linewidth=2, markersize=5)

    ax.set_xlabel("Iteration k")
    ax.set_ylabel("Error $e_k$")
    ax.set_title("Convergence rates on a semilog plot")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=False, loc="lower left")

    # Set a nice y-range (positive only) for readability
    ymin = min(e_lin.min(), e_sup.min(), e_quad.min())
    ax.set_ylim(bottom=max(ymin/10, 1e-16), top=e0*1.2)

    plt.tight_layout()
    plt.savefig("convergence_semilog.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
