"""Run the experiments and write figures/*.png."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bs
import hedging
from hedging import Book, delta_hedge, gbm_paths

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)

S0 = 100.0
K = 100.0
T = 30 / 252
N_PATHS = 20000
N_STEPS = 240  # 8 rehedge opportunities per trading day
SEED = 12345

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

short_call = Book(legs=[("call", K, 1.0)])
short_put = Book(legs=[("put", K, 1.0)])
short_straddle = Book(legs=[("call", K, 1.0), ("put", K, 1.0)])


def exp1_identity():
    """sigma_imp = sigma_real, frequent hedging, no costs -> PnL centred on zero."""
    sigma = 0.20
    paths = gbm_paths(S0, sigma, T, N_STEPS, N_PATHS, seed=SEED)
    res = delta_hedge(paths, short_call, T, sigma, cost=0.0, every=1)
    pnl = res["pnl"]
    premium = res["premium"][0]

    unhedged = premium - short_call.payoff(paths[:, -1])

    se = pnl.std(ddof=1) / np.sqrt(N_PATHS)
    print(f"[1] premium               {premium:8.2f}")
    print(f"[1] hedged   mean {pnl.mean():8.3f}  se {se:6.3f}  sd {pnl.std():7.2f}")
    print(f"[1] unhedged mean {unhedged.mean():8.3f}  sd {unhedged.std():7.2f}")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    bins = np.linspace(-1200, 400, 120)
    ax.hist(unhedged, bins=bins, alpha=0.5, label="unhedged", color="#999999")
    ax.hist(pnl, bins=bins, alpha=0.85, label="delta hedged", color="#1f77b4")
    ax.set_yscale("log")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("P&L per contract")
    ax.set_ylabel("paths (log)")
    ax.set_title(
        f"short one 30d ATM call, $\\sigma_{{imp}}=\\sigma_{{real}}=20\\%$\n"
        f"hedged sd {pnl.std():.1f} vs unhedged sd {unhedged.std():.1f}, "
        f"both means $\\approx$ 0"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/01_identity.png")
    plt.close(fig)


def exp2_mispricing():
    """Sweep sigma_imp against fixed sigma_real."""
    sigma_real = 0.20
    grid = np.arange(0.14, 0.2701, 0.01)
    paths = gbm_paths(S0, sigma_real, T, N_STEPS, N_PATHS, seed=SEED)

    means, ses = [], []
    for s_imp in grid:
        r = delta_hedge(paths, short_call, T, s_imp, cost=0.0, every=1)
        means.append(r["pnl"].mean())
        ses.append(r["pnl"].std(ddof=1) / np.sqrt(N_PATHS))
    means = np.array(means)
    ses = np.array(ses)

    theo = np.array(
        [hedging.theoretical_pnl(short_call, S0, T, s, sigma_real) for s in grid]
    )
    vega_pt = short_call.contract_size * bs.vega(S0, K, T, sigma_real) * 0.01

    print(f"[2] vega per vol point    {vega_pt:8.2f}")
    print(f"[2] fitted slope per pt   {np.polyfit(grid, means, 1)[0] * 0.01:8.2f}")
    print(f"[2] max |mc - theory|     {np.abs(means - theo).max():8.3f}")

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.errorbar(
        grid * 100, means, yerr=1.96 * ses, fmt="o", ms=3.5, lw=1, label="Monte Carlo"
    )
    ax.plot(
        grid * 100,
        theo,
        "-",
        lw=1.2,
        color="#d62728",
        label=r"$V(\sigma_{imp}) - V(\sigma_{real})$",
    )
    ax.axvline(sigma_real * 100, color="k", lw=0.8, ls="--")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\sigma_{imp}$  (%)")
    ax.set_ylabel("mean P&L per contract")
    ax.set_title(
        f"selling at the wrong vol, $\\sigma_{{real}}=20\\%$\n"
        f"slope $\\approx$ vega = {vega_pt:.0f} per vol point"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/02_mispricing.png")
    plt.close(fig)


def exp3_frequency():
    """Cost vs risk as a function of rehedge frequency, across cost levels.

    At a 1bp half-spread the trade-off barely bites: hedging more is almost
    free and always reduces risk. An interior optimum only appears once
    costs are material, so sweep them.
    """
    sigma = 0.20
    paths = gbm_paths(S0, sigma, T, N_STEPS, N_PATHS, seed=SEED)
    everys = [240, 120, 48, 24, 12, 8, 4, 2, 1]
    costs = [1e-4, 1e-3, 5e-3, 2e-2]
    lam = 0.002

    base = {}
    for e in everys:
        r = delta_hedge(paths, short_call, T, sigma, cost=0.0, every=e)
        base[e] = (r["n_trades"].mean(), r["pnl"].std(ddof=1), r["turnover"].mean())

    n = np.array([base[e][0] for e in everys])
    sd = np.array([base[e][1] for e in everys])
    turn = np.array([base[e][2] for e in everys])

    print("[3] rehedges      sd   turnover")
    for a, b, c in zip(n, sd, turn):
        print(f"     {a:8.1f}  {b:6.2f}  {c:9.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))

    ax = axes[0]
    ax.plot(n, sd, "o-", ms=3.5, lw=1.2, label="P&L sd (no costs)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rehedges over the option's life")
    ax.set_ylabel("P&L sd per contract")
    fit = np.polyfit(np.log(n), np.log(sd), 1)[0]
    ax.set_title(f"discretisation error\nslope {fit:.2f} on log-log, theory $-1/2$")
    ax.legend(frameon=False)

    ax = axes[1]
    for c in costs:
        ce = []
        for e in everys:
            r = delta_hedge(paths, short_call, T, sigma, cost=c, every=e)
            ce.append(r["pnl"].mean() - lam * r["pnl"].var(ddof=1))
        ce = np.array(ce)
        best = int(np.argmax(ce))
        ax.plot(n, ce - ce.max(), "o-", ms=3.5, lw=1.2, label=f"{c*1e4:.0f}bp")
        ax.plot([n[best]], [0], "k*", ms=9)
        print(f"[3] cost {c*1e4:5.0f}bp -> optimum at {n[best]:6.1f} rehedges")
    ax.set_xscale("log")
    ax.set_ylim(-120, 12)
    ax.set_xlabel("rehedges over the option's life")
    ax.set_ylabel("certainty equivalent, relative to own best")
    ax.set_title(f"optimum moves left as costs rise\n$\\lambda$={lam}, stars mark the peak")
    ax.legend(frameon=False, title="half-spread", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/03_frequency.png")
    plt.close(fig)


def exp4_band():
    """Band rehedging against fixed-interval rehedging, on a cost/risk frontier."""
    sigma = 0.20
    cost = 1e-4
    paths = gbm_paths(S0, sigma, T, N_STEPS, N_PATHS, seed=SEED)

    fixed = []
    for e in [120, 48, 24, 12, 8, 4, 2, 1]:
        r = delta_hedge(paths, short_call, T, sigma, cost=cost, every=e)
        fixed.append((r["cost_paid"].mean(), r["pnl"].std(ddof=1), r["n_trades"].mean()))

    banded = []
    for b in [40, 25, 15, 10, 6, 4, 2.5, 1.5]:
        r = delta_hedge(paths, short_call, T, sigma, cost=cost, every=1, band=b)
        banded.append((r["cost_paid"].mean(), r["pnl"].std(ddof=1), r["n_trades"].mean(), b))

    fx = np.array(fixed)
    bd = np.array(banded)

    print("[4] fixed  cost/sd:", ", ".join(f"{c:.1f}/{s:.1f}" for c, s, _ in fixed))
    print("[4] band   cost/sd:", ", ".join(f"{c:.1f}/{s:.1f}" for c, s, _, _ in banded))

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(fx[:, 0], fx[:, 1], "o-", ms=4, lw=1.2, label="fixed interval")
    ax.plot(bd[:, 0], bd[:, 1], "s-", ms=4, lw=1.2, color="#d62728", label="delta band")
    for c, s, _, b in banded[::2]:
        ax.annotate(f"{b:g}", (c, s), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_xlabel("mean transaction cost paid per contract")
    ax.set_ylabel("P&L sd per contract")
    ax.set_title("band rehedging dominates at equal cost\n(labels: band width in shares)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/04_band.png")
    plt.close(fig)


def exp5_straddle():
    """Delta cancels between call and put; gamma and vega add."""
    books = [("short call", short_call), ("short put", short_put), ("short straddle", short_straddle)]

    print("[5]            delta    gamma     vega/pt")
    for name, bk in books:
        print(
            f"     {name:15s} {bk.delta(S0, T, 0.20):7.1f} "
            f"{bk.gamma(S0, T, 0.20):8.2f} {bk.vega(S0, T, 0.20) * 0.01:8.1f}"
        )

    grid = np.arange(0.12, 0.2901, 0.02)
    curves = {name: [] for name, _ in books}
    for s_real in grid:
        paths = gbm_paths(S0, s_real, T, N_STEPS, N_PATHS, seed=SEED)
        for name, bk in books:
            r = delta_hedge(paths, bk, T, 0.20, cost=0.0, every=1)
            curves[name].append(r["pnl"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))

    ax = axes[0]
    for name, bk in books:
        S = np.linspace(85, 115, 200)
        ax.plot(S, bk.delta(S, T, 0.20), lw=1.4, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("spot")
    ax.set_ylabel("position delta (shares)")
    ax.set_title("delta: the straddle sits at zero")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for name, _ in books:
        y = np.array(curves[name])
        slope = np.polyfit(grid, y, 1)[0] * 0.01
        ax.plot(grid * 100, y, "o-", ms=3.5, lw=1.2, label=f"{name}  ({slope:.0f}/pt)")
        print(f"[5] {name:15s} slope per vol point {slope:7.1f}")
    ax.axvline(20, color="k", lw=0.8, ls="--")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\sigma_{real}$  (%)")
    ax.set_ylabel("mean P&L")
    ax.set_title(r"sensitivity to realised vol, all sold at $\sigma_{imp}=20\%$")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/05_straddle.png")
    plt.close(fig)


if __name__ == "__main__":
    exp1_identity()
    exp2_mispricing()
    exp3_frequency()
    exp4_band()
    exp5_straddle()
    print("figures written to", FIG)
