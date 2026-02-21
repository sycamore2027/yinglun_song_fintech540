from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, t


def read_csv_clean(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    return df


def find_file(base_dir, name, *fallbacks):
    for path in (base_dir / name, *fallbacks):
        path = Path(path)
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {name}")


def var_es(x, alpha=0.05):
    x = np.sort(np.asarray(x, dtype=float))
    q = np.quantile(x, alpha, method="midpoint")
    return -float(q), -float(x[x <= q].mean())


def summarize_risk(values):
    first_iter = values["iteration"].min()
    rows = []

    for stock, g in values.groupby("Stock", sort=False):
        pnl_iter = g.groupby("iteration", sort=False)["pnl"].sum().to_numpy()
        current = float(g.loc[g["iteration"].eq(first_iter), "currentValue"].sum())
        var95, es95 = var_es(pnl_iter)
        rows.append(
            {
                "Stock": stock,
                "VaR95": var95,
                "ES95": es95,
                "VaR95_Pct": var95 / current,
                "ES95_Pct": es95 / current,
            }
        )

    total_pnl = values.groupby("iteration", sort=False)["pnl"].sum().to_numpy()
    total_current = float(values.loc[values["iteration"].eq(first_iter), "currentValue"].sum())
    var95, es95 = var_es(total_pnl)
    rows.append(
        {
            "Stock": "Total",
            "VaR95": var95,
            "ES95": es95,
            "VaR95_Pct": var95 / total_current,
            "ES95_Pct": es95 / total_current,
        }
    )
    return pd.DataFrame(rows)[["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"]]


def main():
    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parents[1]

    returns = read_csv_clean(find_file(base_dir, "test9_1_returns.csv"))
    portfolio_in = read_csv_clean(
        find_file(
            base_dir,
            "test9_1_portfolio.csv",
            repo_root / "FinTech-545-Fall2025/testfiles/data/test9_1_portfolio.csv",
        )
    )

    a = returns["A"].to_numpy(dtype=float)
    b = returns["B"].to_numpy(dtype=float)
    mu_a, sigma_a = norm.fit(a)
    nu_b, mu_b, sigma_b = t.fit(b)
    dist_a = norm(loc=mu_a, scale=max(sigma_a, 1e-12))
    dist_b = t(df=nu_b, loc=mu_b, scale=max(sigma_b, 1e-12))

    u = np.column_stack([dist_a.cdf(a), dist_b.cdf(b)])
    rho = float(spearmanr(u, axis=0).statistic)
    corr = np.array([[1.0, rho], [rho, 1.0]])

    n_sim = 100000
    rng = np.random.default_rng(1234)
    z = rng.multivariate_normal(mean=[0.0, 0.0], cov=corr, size=n_sim)
    u_sim = norm.cdf(z)
    sim_ret = np.column_stack([dist_a.ppf(u_sim[:, 0]), dist_b.ppf(u_sim[:, 1])])

    if {"Stock", "Holding", "Starting Price"}.issubset(portfolio_in.columns):
        portfolio = portfolio_in.assign(
            currentValue=lambda d: d["Holding"].astype(float) * d["Starting Price"].astype(float)
        )[["Stock", "currentValue"]]
    else:
        portfolio = pd.DataFrame({"Stock": ["A", "B"], "currentValue": [2000.0, 3000.0]})

    values = portfolio.merge(pd.DataFrame({"iteration": np.arange(n_sim)}), how="cross")
    stock_col = values["Stock"].map({"A": 0, "B": 1}).to_numpy()
    sim_col = values["iteration"].to_numpy()
    current = values["currentValue"].to_numpy(dtype=float)
    simulated = current * (1.0 + sim_ret[sim_col, stock_col])

    values["simulatedValue"] = simulated
    values["pnl"] = simulated - current

    summarize_risk(values).to_csv(base_dir / "testout9_1.csv", index=False)

if __name__ == "__main__":
    main()
