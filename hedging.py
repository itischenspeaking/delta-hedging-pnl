"""GBM paths and discrete delta hedging of a short option position.

Convention: we are short the option(s), hedged with the underlying.
Cash account, no interest (r = 0). Transaction costs are a proportional
half-spread paid on the stock leg only.
"""

from dataclasses import dataclass, field

import numpy as np

import bs


def gbm_paths(S0, sigma, T, n_steps, n_paths, seed=0):
    """Return array of shape (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_incr = -0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_incr, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)


@dataclass
class Book:
    """A short position in one or more options on the same underlying.

    legs: list of (kind, strike, quantity). Positive quantity means short
    that many contracts, since the whole module is written from the
    market maker's side.
    """

    legs: list = field(default_factory=lambda: [("call", 100.0, 1.0)])
    contract_size: float = 100.0

    def premium(self, S, tau, sigma):
        return self.contract_size * sum(
            q * bs.price(S, K, tau, sigma, kind) for kind, K, q in self.legs
        )

    def delta(self, S, tau, sigma):
        """Delta of the short position, in shares."""
        return -self.contract_size * sum(
            q * bs.delta(S, K, tau, sigma, kind) for kind, K, q in self.legs
        )

    def gamma(self, S, tau, sigma):
        return -self.contract_size * sum(
            q * bs.gamma(S, K, tau, sigma) for _, K, q in self.legs
        )

    def vega(self, S, tau, sigma):
        return -self.contract_size * sum(
            q * bs.vega(S, K, tau, sigma) for _, K, q in self.legs
        )

    def payoff(self, S):
        """Cash paid out at expiry (positive number = we owe it)."""
        out = np.zeros_like(np.asarray(S, dtype=float))
        for kind, K, q in self.legs:
            if kind == "call":
                out = out + q * np.maximum(S - K, 0.0)
            else:
                out = out + q * np.maximum(K - S, 0.0)
        return self.contract_size * out


def delta_hedge(
    paths,
    book,
    T,
    sigma_imp,
    cost=0.0,
    every=1,
    band=None,
):
    """Sell the book at sigma_imp, then delta hedge along each path.

    paths     (n_paths, n_steps + 1) underlying
    sigma_imp volatility used for both pricing and hedging
    cost      proportional half-spread on the stock leg (e.g. 1e-4)
    every     rehedge every `every` steps
    band      if set, rehedge only when the hedge error in shares exceeds
              this; overrides `every` as the trigger but is still only
              checked on the `every` grid

    Returns dict with total pnl per path and diagnostics.
    """
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    dt = T / n_steps

    premium = book.premium(paths[:, 0], T, sigma_imp)
    cash = premium.astype(float).copy()

    target = book.delta(paths[:, 0], T, sigma_imp)
    held = -target  # hedge position in shares, offsetting the book delta
    cash -= held * paths[:, 0]
    turnover = np.abs(held) * paths[:, 0]
    cash -= cost * turnover
    n_trades = np.ones(n_paths)

    for i in range(1, n_steps):
        if i % every:
            continue
        S = paths[:, i]
        tau = T - i * dt
        want = -book.delta(S, tau, sigma_imp)
        gap = want - held
        if band is None:
            trade = gap
        else:
            trade = np.where(np.abs(gap) > band, gap, 0.0)
        traded = np.abs(trade) > 0
        cash -= trade * S
        cash -= cost * np.abs(trade) * S
        turnover += np.abs(trade) * S
        n_trades += traded
        held = held + trade

    S_T = paths[:, -1]
    cash += held * S_T
    cash -= cost * np.abs(held) * S_T
    turnover += np.abs(held) * S_T
    cash -= book.payoff(S_T)

    return {
        "pnl": cash,
        "premium": premium,
        "turnover": turnover,
        "n_trades": n_trades,
        "cost_paid": cost * turnover,
    }


def theoretical_pnl(book, S0, T, sigma_imp, sigma_real):
    """Expected P&L when hedging at implied vol: V(sigma_imp) - V(sigma_real)."""
    return book.premium(S0, T, sigma_imp) - book.premium(S0, T, sigma_real)
