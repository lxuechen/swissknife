"""
[x] terminate based on duality gap at 1e-3
[ ] check against cvxpy
[ ] experiment with mu; inner newton steps and total newton steps
[ ] plot log duality gap vs total newton steps
    - textbook format step plot

python -m ee364a.a11_8
"""

import fire
import torch

from .a10_4 import Soln, LPCenteringProb, infeasible_start_newton_solve


def barrier_solve(soln: Soln, prob: LPCenteringProb, t: float, mu: float, epsilon=1e-3):
    this_step = 0
    steps = []
    newton_steps = []
    gaps = []
    while True:
        prob.t = t  # Solve the right problem.
        soln, this_newton_steps, _, _, _ = infeasible_start_newton_solve(
            soln=soln, prob=prob,
        )

        this_step += 1
        this_gap = prob.m / t
        steps.append(this_step)
        gaps.append(this_gap)
        newton_steps.append(this_newton_steps[-1])

        if this_gap < epsilon:
            break
        t = mu * t

    return soln, steps, gaps, newton_steps


def _generate_prob():
    m = 100
    n = 500

    A = torch.randn(m, n)
    A[0].abs_()
    rank = torch.linalg.matrix_rank(A)
    assert rank == m

    p = torch.randn(n).abs()  # Make positive.
    b = A @ p
    c = torch.randn(n)
    in_domain = lambda soln: torch.all(soln.x > 0)

    x = torch.randn(n).exp() * 3  # Make positive.
    nu = torch.zeros(m)

    return Soln(x=x, nu=nu), LPCenteringProb(A=A, b=b, c=c, in_domain=in_domain)


def main():
    soln, prob = _generate_prob()

    soln, steps, gaps, newton_steps = barrier_solve(
        soln=soln, prob=prob, t=1, mu=2
    )
    print(gaps)
    print(newton_steps)


if __name__ == "__main__":
    fire.Fire(main)
