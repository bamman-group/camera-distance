from pathlib import Path

import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import brentq

SCRIPT_DIR = Path(__file__).resolve().parent


START_YEARS = {
    "SOUND":      1929,
    "TELEVISION": 1951,
    "SMARTPHONE": 2012,
}

ANCHOR_OFFSET = 1
HALF_WIDTH    = 1
HORIZON       = 10

CATEGORIES = ["MCU", "M", "ML", "L"]

BENCHMARKS = {"MCU": 0.05, "M": 0.05, "ML": 0.05, "L": 0.05}

ALPHA        = 0.05
N_PERMUTE    = 10_000
RANDOM_SEED  = 1

COLLECTION = "popular"
COLLECTION_FLAG = {"popular": "is_popular", "prestige": "is_prestige", "indie": "is_indie"}

FILM_LEVEL_CSV = SCRIPT_DIR / ".." / "data" / "analysis_outputs" / "film_level_categories.csv"
INTERMEDIATE_DIR = SCRIPT_DIR / ".." / "data" / "analysis_outputs"


SCALE_COLS = ["XCU", "CU", "MCU", "M", "ML", "L", "XLS"]


def load(path=FILM_LEVEL_CSV):
    df = pd.read_csv(path)
    df = df[df[COLLECTION_FLAG[COLLECTION]] == 1].copy()
    df = df.rename(columns={c.lower(): c for c in SCALE_COLS})

    total = df[SCALE_COLS].sum(axis=1)
    df = df[total > 0].copy()
    for c in SCALE_COLS:
        df[c] = df[c] / total[total > 0]
    return df


def window(df, center):
    return df[(df.year - center).abs() <= HALF_WIDTH]


def power_two_sample(delta, sd, n_per_group, alpha=None):
    alpha = ALPHA if alpha is None else alpha
    dof = 2 * n_per_group - 2
    ncp = delta / (sd * np.sqrt(2 / n_per_group))
    crit = stats.t.ppf(1 - alpha / 2, dof)
    with np.errstate(all="ignore"):
        val = stats.nct.sf(crit, dof, ncp) + stats.nct.cdf(-crit, dof, ncp)
    if not np.isfinite(val):
        val = 1.0 if ncp > crit else 0.0
    return float(np.clip(val, 0.0, 1.0))


def minimum_detectable_effect(sd, n_per_group, power, alpha=None):
    f = lambda d: power_two_sample(d, sd, n_per_group, alpha) - power
    return brentq(f, 1e-9, 10 * sd)


def permutation_p(x, y, rng):
    obs = abs(y.mean() - x.mean())
    pooled, n_x = np.concatenate([x, y]), len(x)
    count = 0
    for _ in range(N_PERMUTE):
        rng.shuffle(pooled)
        if abs(pooled[n_x:].mean() - pooled[:n_x].mean()) >= obs - 1e-12:
            count += 1
    return (count + 1) / (N_PERMUTE + 1)


def main():
    df = load()
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for name, start in START_YEARS.items():
        anchor = start - ANCHOR_OFFSET
        pre, post = window(df, anchor), window(df, anchor + HORIZON)
        for cat in CATEGORIES:
            x, y = pre[cat].to_numpy(), post[cat].to_numpy()
            t = stats.ttest_ind(y, x, equal_var=False)
            ci = t.confidence_interval(1 - ALPHA)
            rows.append(dict(
                event=name, cat=cat, anchor=anchor,
                pre_lo=anchor - HALF_WIDTH, pre_hi=anchor + HALF_WIDTH,
                post_lo=anchor + HORIZON - HALF_WIDTH,
                post_hi=anchor + HORIZON + HALF_WIDTH,
                n_pre=len(x), n_post=len(y),
                mean_pre=x.mean(), mean_post=y.mean(),
                diff=y.mean() - x.mean(), lo=ci.low, hi=ci.high,
                p=t.pvalue, perm_p=permutation_p(x, y, rng),
                sd=np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2),
                se=(y.mean() - x.mean() - ci.low) / stats.t.ppf(1 - ALPHA / 2, t.df),
                dof=t.df, n_eff=2 / (1 / len(x) + 1 / len(y)),
            ))

    res = pd.DataFrame(rows)

    n_tests = len(START_YEARS) * len(CATEGORIES)
    alpha_bonf = ALPHA / n_tests
    res["p_bonf"] = np.minimum(res.p * n_tests, 1.0)
    res["perm_p_bonf"] = np.minimum(res.perm_p * n_tests, 1.0)

    print(f"\n{'='*104}")
    print(f"TEN-YEAR CHANGE IN SHOT DISTANCE  "
          f"(anchor = start - {ANCHOR_OFFSET}, windows +/-{HALF_WIDTH} yr, "
          f"horizon {HORIZON} yr, alpha = {ALPHA})")
    print("start years: " + ", ".join(f"{k} {v}" for k, v in START_YEARS.items()))
    print(f"Bonferroni correction: m={n_tests} tests, adjusted alpha={alpha_bonf:.5f}")
    print(f"{'='*104}")
    print(f"{'event':<11} {'cat':<4} {'pre window':<10} {'n':>4}  "
          f"{'post window':<11} {'n':>4}  {'before':>7} {'after':>7} "
          f"{'diff':>8}  {'95% CI':>17} {'p':>9} {'p_bonf':>9} {'perm':>7} {'perm_bonf':>9}  sig")
    for r in res.itertuples():
        print(f"{r.event:<11} {r.cat:<4} {r.pre_lo}-{r.pre_hi:<5} {r.n_pre:>4}  "
              f"{r.post_lo}-{r.post_hi:<6} {r.n_post:>4}  "
              f"{r.mean_pre:>7.3f} {r.mean_post:>7.3f} {r.diff:>+8.3f}  "
              f"[{r.lo:+.3f},{r.hi:+.3f}] {r.p:>9.2e} {r.p_bonf:>9.2e} "
              f"{r.perm_p:>7.4f} {r.perm_p_bonf:>9.4f}  "
              f"{'YES' if r.p_bonf < ALPHA else 'no'}")

    ns = res[res.p_bonf >= ALPHA]
    print(f"\n{'='*104}")
    print("POWER AND EQUIVALENCE FOR NON-SIGNIFICANT CELLS")
    print(f"{'='*104}")
    print(f"{'event':<11} {'cat':<4} {'est':>8}  {'95% CI':>17} "
          f"{'MDE80':>7} {'MDE90':>7} {'power':>7} {'bound':>7}  TOST vs benchmark")
    equiv_rows = []
    for r in ns.itertuples():
        delta = BENCHMARKS[r.cat]
        mde80 = minimum_detectable_effect(r.sd, r.n_eff, 0.80)
        mde90 = minimum_detectable_effect(r.sd, r.n_eff, 0.90)
        power = power_two_sample(delta, r.sd, r.n_eff)
        bound = abs(r.diff) + stats.t.ppf(1 - ALPHA, r.dof) * r.se
        tost = max(stats.t.cdf((r.diff - delta) / r.se, r.dof),
                   1 - stats.t.cdf((r.diff + delta) / r.se, r.dof))
        verdict = "REJECT" if tost < ALPHA else "cannot rule out"
        print(f"{r.event:<11} {r.cat:<4} {r.diff:>+8.4f}  "
              f"[{r.lo:+.4f},{r.hi:+.4f}] {mde80:>7.4f} {mde90:>7.4f} "
              f"{power:>7.3f} {bound:>7.4f}  H0:|eff|>={delta:.3f}  "
              f"p={tost:.3g}  {verdict}")
        equiv_rows.append(dict(
            event=r.event, cat=r.cat, diff=r.diff, lo=r.lo, hi=r.hi,
            benchmark=delta, mde80=mde80, mde90=mde90, power=power,
            bound=bound, tost_p=tost, rejects_benchmark=tost < ALPHA,
        ))

    print("\nbenchmarks: " + ", ".join(f"{k} {v*100:.1f}pp"
                                       for k, v in BENCHMARKS.items()))
    print()

    out_dir = INTERMEDIATE_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tests_path = out_dir / "camera_distance_tests.csv"
    equiv_path = out_dir / "camera_distance_equivalence.csv"
    res.assign(alpha=ALPHA, n_tests=n_tests,
               significant=res.p_bonf < ALPHA).to_csv(tests_path, index=False)
    pd.DataFrame(equiv_rows, columns=[
        "event", "cat", "diff", "lo", "hi", "benchmark", "mde80", "mde90",
        "power", "bound", "tost_p", "rejects_benchmark",
    ]).to_csv(equiv_path, index=False)
    print(f"Wrote {tests_path}")
    print(f"Wrote {equiv_path}")


if __name__ == "__main__":
    main()
