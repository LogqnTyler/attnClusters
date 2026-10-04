import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import torch
    from attention import Attention, Attn_1D, ForgetfulAttention, device
    from torch.linalg import matrix_rank
    from pathlib import Path
    from sklearn.datasets import make_spd_matrix

    torch.set_default_dtype(torch.float64)
    torch.set_default_device(device)
    return (
        ForgetfulAttention,
        Path,
        make_spd_matrix,
        matrix_rank,
        mo,
        np,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Goals:
    We will investigating the asymptotic bahavior of the token clustering in three cases
    1. $Q, K, V = I_3$,
    2. $Q^T K \succ 0, V = I_3$, and
    3. $Q, K$ are random matrices, but $V = I_3$.

    The paper (GLPR 23) proves that under cases 1,2, tokens form clusters on the boundry (usually vertices) of a convex polytope. Here we try and explore the asymptotic behavior of the number of vetices a convex polytope has based on the number of initial tokens. We start our investigation in dimension 3.
    """)
    return


@app.cell
def _(torch):
    D = 3
    V = torch.eye(D)
    N_trials = 1500
    Ns = [5, 10, 50, 100, 500, 1000, 5000]
    seed = 42
    return D, N_trials, Ns, V


@app.cell(hide_code=True)
def rank_histogram_helper(Ns, Path, mo, np):
    def makeRankHistogram(filename, title):
        import matplotlib.pyplot as plt

        path = Path(filename)
        if not path.exists():
            return mo.md(f"Run the rank computation to create `{filename}`.")

        ranks = np.load(path)
        tokenCounts = Ns[: ranks.shape[0]]
        columns = 2
        rows = int(np.ceil(len(tokenCounts) / columns))
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(12, 4 * rows),
            squeeze=False,
        )

        for index, N in enumerate(tokenCounts):
            row = index // columns
            column = index % columns
            axis = axes[row, column]
            rankValues = ranks[index]
            rankMin = int(np.nanmin(rankValues))
            rankMax = int(np.nanmax(rankValues))
            bins = np.arange(rankMin - 0.5, rankMax + 1.5, 1)

            axis.hist(
                rankValues,
                bins=bins,
                color="#4c72b0",
                edgecolor="white",
                linewidth=0.8,
                rwidth=0.9,
            )
            axis.set_title(f"N = {N}")
            axis.set_xlabel("Final attention rank")
            axis.set_ylabel("Frequency")
            axis.set_xticks(range(rankMin, rankMax + 1))
            axis.grid(axis="y", alpha=0.2)

        for index in range(len(tokenCounts), rows * columns):
            axes.flat[index].set_visible(False)

        figure.suptitle(title, fontsize=15)
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        return figure

    return (makeRankHistogram,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1: $Q, K, V = I_3$
    """)
    return


@app.cell
def _(D, ForgetfulAttention, N_trials, Ns, Path, V, matrix_rank, np, torch):
    def computeRanksPart1():
        if not Path("ranks_case1.npy").exists():
            Q = torch.eye(D)
            K = torch.eye(D)
            ranks = np.zeros((len(Ns), N_trials))
            for i in range(len(Ns)):
                N = Ns[i]
                for trial in range(N_trials):
                    attn = ForgetfulAttention(
                        K, Q, V, X=torch.rand(D, N) * 10 - 5
                    )
                    attn.rescaled_dynamics()
                    ranks[i, trial] = matrix_rank(attn.P).item()
            np.save("ranks_case1.npy", ranks)
        else:
            ranks = np.load("ranks_case1.npy")

    computeRanksPart1()
    return


@app.cell(hide_code=True)
def part_1_rank_histogram(makeRankHistogram):
    part1RankHistogram = makeRankHistogram(
        "ranks_case1.npy",
        "Part 1: Final Attention-Rank Distributions",
    )
    part1RankHistogram
    return


@app.cell(hide_code=True)
def part_2_header(mo):
    mo.md(r"""
    ## Part 2: $Q^T K \succ 0$, $V = I_3$

    For $Q^T K \succ 0$, $Q^T K$ must be invertible, which implies that $Q, K$ are invertible as well. So we sample $Q$ uniformly on $[-1,1]$ with a bounded condition number $\kappa(Q) <= 10^4$ ensuring that $Q$ will be practically invertible. Then we sample a SPD matrix $M$ and define $K = Q^{-T} M$ so that $Q^T K = Q^T Q^{-T} M = M \succ 0.$
    """)
    return


@app.cell
def part_2_ranks(
    D,
    ForgetfulAttention,
    N_trials,
    Ns,
    Path,
    V,
    make_spd_matrix,
    matrix_rank,
    np,
    torch,
):
    def sampleQK(D):
        while True:
            Q = torch.rand((D, D)) * 2 - 1
            if torch.linalg.cond(Q) < 1e4:
                break
        while True:
            M = torch.tensor(make_spd_matrix(D))
            if torch.linalg.cond(M) < 1e4:
                break
        return Q, torch.linalg.inv(Q.T) @ M

    def computeRanksPart2():
        if not Path("ranks_case2.npy").exists():
            Q, K = sampleQK(D)
            ranks = np.zeros((len(Ns), N_trials))
            for i in range(len(Ns)):
                N = Ns[i]
                for trial in range(N_trials):
                    attn = ForgetfulAttention(
                        K, Q, V, X=torch.rand(D, N) * 10 - 5
                    )
                    attn.rescaled_dynamics()
                    ranks[i, trial] = matrix_rank(attn.P).item()
            np.save("ranks_case2.npy", ranks)
        else:
            ranks = np.load("ranks_case2.npy")

    computeRanksPart2()
    return


@app.cell(hide_code=True)
def part_2_rank_histogram(makeRankHistogram):
    part2RankHistogram = makeRankHistogram(
        "ranks_case2.npy",
        "Part 2: Final Attention-Rank Distributions",
    )
    part2RankHistogram
    return


@app.cell(hide_code=True)
def part_3_header(mo):
    mo.md(r"""
    ## Part 3: Random $Q, K$, $V = I_3$
    """)
    return


@app.cell
def part_3_ranks(
    D,
    ForgetfulAttention,
    N_trials,
    Ns,
    Path,
    V,
    matrix_rank,
    np,
    torch,
):
    def computeRanksPart3():
        if not Path("ranks_case3.npy").exists():
            Q = torch.rand(D, D) * 2 - 1
            K = torch.rand(D, D) * 2 - 1
            ranks = np.zeros((len(Ns), N_trials))
            for i in range(len(Ns)):
                N = Ns[i]
                for trial in range(N_trials):
                    attn = ForgetfulAttention(
                        K, Q, V, X=torch.rand(D, N) * 10 - 5
                    )
                    attn.rescaled_dynamics()
                    ranks[i, trial] = matrix_rank(attn.P).item()
            np.save("ranks_case3.npy", ranks)
        else:
            ranks = np.load("ranks_case3.npy")

    computeRanksPart3()
    return


@app.cell(hide_code=True)
def part_3_rank_histogram(makeRankHistogram):
    part3RankHistogram = makeRankHistogram(
        "ranks_case3.npy",
        "Part 3: Final Attention-Rank Distributions",
    )
    part3RankHistogram
    return


if __name__ == "__main__":
    app.run()
