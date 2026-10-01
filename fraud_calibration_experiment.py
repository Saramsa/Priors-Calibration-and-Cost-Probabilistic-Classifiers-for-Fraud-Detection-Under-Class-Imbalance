#!/usr/bin/env python3
"""
Priors, Calibration, and Cost: probabilistic classifiers for fraud detection under class imbalance.

Reproduces every number, Table I, and Fig. 1 in the paper.

Usage:
    python fraud_calibration_experiment.py --data path/to/creditcard.csv --out results/

Requires: numpy, pandas, scikit-learn, matplotlib.
Data: ULB credit card fraud dataset (Kaggle: mlg-ulb/creditcardfraud), 284,807 rows, column "Class" = label.

Outputs (in --out):
    results_raw.csv          one row per split x model x condition
    results_sensitivity.csv  costs at other c_FN / c_FP ratios
    table1.csv               Table I as "mean (sd)" text
    fig1_reliability.png     Fig. 1 (300 dpi)

Conditions:  A raw | B undersampled | C undersampled + prior correction, Eq. (3) | D raw + Platt scaling
Costs are assumed: c_FP = 5, c_FN = 100, so t* = c_FP / (c_FP + c_FN) = 0.0476, Eq. (5).
"""
import argparse, os, time, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")

C_FP, C_FN = 5.0, 100.0
T_STAR = C_FP / (C_FP + C_FN)
RATIOS = [5, 20, 100]                      # c_FN / c_FP for the sensitivity analysis (c_FP fixed at 5)
NAMES = {"A": "A: raw", "B": "B: undersampled", "C": "C: undersampled + corrected", "D": "D: raw + Platt"}


def correct(p_s, beta):
    """Eq. (3): recover the true posterior from a model trained on a sample that keeps a fraction beta of negatives."""
    return beta * p_s / (beta * p_s - p_s + 1.0)


def cost(y, p, t, c_fp=C_FP, c_fn=C_FN):
    """Expected cost per transaction when flagging cases with p > t."""
    flag = p > t
    fp = np.sum(flag & (y == 0))
    fn = np.sum((~flag) & (y == 1))
    return (c_fp * fp + c_fn * fn) / len(y)


def make_model(name):
    if name == "NB":
        return GaussianNB()
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def run(X, y, n_seeds):
    rows, sens, preds, sanity = [], [], {}, []
    t0 = time.time()
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
        beta = len(pos) / len(neg)                         # fraction of legitimate cases kept
        idx = np.concatenate([pos, rng.choice(neg, size=len(pos), replace=False)])
        Xus, yus = Xtr[idx], ytr[idx]
        pi_tr = ytr.mean()
        base_rate = yte.mean()
        for name in ["NB", "LR"]:
            for cond in "ABCD":
                if cond == "A":
                    p = make_model(name).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                elif cond == "D":
                    m = CalibratedClassifierCV(make_model(name), method="sigmoid", cv=5).fit(Xtr, ytr)
                    p = m.predict_proba(Xte)[:, 1]
                else:
                    p_s = make_model(name).fit(Xus, yus).predict_proba(Xte)[:, 1]
                    p = correct(p_s, beta) if cond == "C" else p_s
                    if cond == "C" and name == "NB":
                        # sanity check: Eq. (3) must equal refitting Gaussian NB with the true training prior
                        g = GaussianNB(priors=[1 - pi_tr, pi_tr]).fit(Xus, yus)
                        sanity.append(np.max(np.abs(g.predict_proba(Xte)[:, 1] - p)))
                rows.append(dict(
                    seed=seed, model=name, cond=cond,
                    auprc=average_precision_score(yte, p), auroc=roc_auc_score(yte, p),
                    brier=brier_score_loss(yte, p), brier_base=base_rate * (1 - base_rate),
                    mean_p=p.mean(), base_rate=base_rate,
                    cost_05=cost(yte, p, 0.5), cost_star=cost(yte, p, T_STAR),
                    flag_05=(p > 0.5).mean(), flag_star=(p > T_STAR).mean(),
                    cost_none=C_FN * base_rate, cost_all=C_FP * (1 - base_rate)))
                for k in RATIOS:
                    t = 1.0 / (1.0 + k)
                    sens.append(dict(
                        seed=seed, model=name, cond=cond, ratio=k,
                        cost_star=cost(yte, p, t, C_FP, C_FP * k),
                        cost_05=cost(yte, p, 0.5, C_FP, C_FP * k),
                        cost_none=C_FP * k * base_rate))
                preds.setdefault((name, cond), []).append((yte.copy(), p.copy()))
        print(f"split {seed + 1}/{n_seeds} done ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(sens), preds, max(sanity)


def make_table(res):
    def f(g, col, scale, nd):
        return f"{g[col].mean() * scale:.{nd}f} ({g[col].std() * scale:.{nd}f})"
    out = []
    for m in ["NB", "LR"]:
        for c in "ABCD":
            g = res[(res.model == m) & (res.cond == c)]
            out.append([m, NAMES[c], f(g, "auprc", 1, 3), f(g, "auroc", 1, 3), f(g, "brier", 1e3, 2),
                        f(g, "cost_05", 1e3, 1), f(g, "cost_star", 1e3, 1)])
    return pd.DataFrame(out, columns=["Model", "Condition", "AUC-PR", "AUC-ROC", "Brier x1e3",
                                      "Cost@0.5 /1000", "Cost@t* /1000"])


def reliability(preds, m, c, base_rate, n_bins=6, floor=1e-6):
    """Pool predictions over all splits; bins hold equal numbers of frauds."""
    y = np.concatenate([a for a, _ in preds[(m, c)]])
    p = np.concatenate([b for _, b in preds[(m, c)]])
    o = np.argsort(p, kind="mergesort")
    y, p = y[o], p[o]
    cum = np.cumsum(y)
    tot = cum[-1]
    edges = [0] + [int(np.searchsorted(cum, tot * k / n_bins)) + 1 for k in range(1, n_bins)] + [len(y)]
    mp, obs = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            mp.append(max(p[a:b].mean(), floor))
            obs.append(y[a:b].mean())
    return np.array(mp), np.array(obs)


def make_figure(preds, base_rate, path, floor=1e-6):
    plt.rcParams.update({"font.size": 7, "font.family": "serif"})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    style = {"A": ("o", "-"), "B": ("s", "--"), "C": ("^", "-"), "D": ("D", ":")}
    panels = [("LR", "ABC", "(a) Logistic regression"), ("NB", "ABCD", "(b) Gaussian Naive Bayes")]
    for a, (m, conds, title) in zip(ax, panels):
        for c in conds:
            mp, obs = reliability(preds, m, c, base_rate)
            mk, ls = style[c]
            a.plot(mp, obs, marker=mk, ls=ls, ms=3.5, lw=1, label=NAMES[c])
        a.plot([floor, 1], [floor, 1], "k--", lw=0.7, label="perfect calibration")
        a.axhline(base_rate, color="gray", lw=0.5, ls=":")
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlim(5e-7, 2); a.set_ylim(5e-5, 1.5)
        a.set_xlabel("Mean predicted fraud probability (floored at $10^{-6}$)")
        a.set_title(title, fontsize=7.5)
        a.grid(alpha=0.25, lw=0.4)
        a.legend(fontsize=6, loc="upper left", frameon=False)
    ax[0].set_ylabel("Observed fraud rate")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to creditcard.csv")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--seeds", type=int, default=10, help="number of random splits (paper uses 10)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.data)
    X, y = df.drop(columns=["Class"]).values, df["Class"].values
    print(f"data: {len(df):,} rows, {int(y.sum())} frauds ({y.mean():.4%}), {X.shape[1]} features")
    if len(df) != 284807 or int(y.sum()) != 492:
        print("WARNING: counts differ from the published ULB dataset (284,807 rows, 492 frauds).")

    res, sens, preds, max_gap = run(X, y, a.seeds)
    res.to_csv(os.path.join(a.out, "results_raw.csv"), index=False)
    sens.to_csv(os.path.join(a.out, "results_sensitivity.csv"), index=False)
    table = make_table(res)
    table.to_csv(os.path.join(a.out, "table1.csv"), index=False)
    make_figure(preds, res.base_rate.mean(), os.path.join(a.out, "fig1_reliability.png"))

    print(f"\nEq. (3) check, max |corrected - refit with true prior| (Gaussian NB): {max_gap:.2e}")
    print("\nTable I (Brier x1e3, costs per 1,000 transactions):")
    print(table.to_string(index=False))
    print(f"\nReference: base-rate Brier x1e3 = {res.brier_base.mean() * 1e3:.2f}; "
          f"flag-none cost = {res.cost_none.mean() * 1e3:.1f}; flag-all cost = {res.cost_all.mean() * 1e3:.0f}")
    print(f"\nFiles written to: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
