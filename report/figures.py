"""Figures du rapport et des slides, depuis checkpoints/ et results/ (rien n'est réentraîné).

    python report/figures.py            # écrit report/figures/*.pdf (LaTeX) et *.png (slides)
"""
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")      # même palette que rl/plot.py, ordre fixe
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STEPS_PER_EPISODE = 100
BIG_STEPS_PER_EPISODE = 400          # big : 400 pas par épisode
ROBOTS = (2, 3, 4)

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.axisbelow": True,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / name)


def qtable_rows():
    """results/qtable_table.md -> {n: {colonne: valeur}}."""
    lines = [l for l in (ROOT / "results" / "qtable_table.md").read_text(encoding="utf-8").splitlines()
             if l.startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = {}
    for line in lines[2:]:
        cells = dict(zip(head, (c.strip() for c in line.strip("|").split("|"))))
        rows[int(cells["Robots"])] = cells
    return rows


def mean_std(cell):
    m, s = re.findall(r"[\d.]+", cell)[:2]
    return float(m), float(s)


def curve(folder, column, window):
    df = pd.read_csv(ROOT / "checkpoints" / folder / "train_log.csv", usecols=["episode", column])
    y = df[column].rolling(window, min_periods=window).mean()
    step = max(1, len(df) // 600)             # ~600 points par courbe suffisent
    return (df["episode"] * STEPS_PER_EPISODE / 1e6)[::step], y[::step]


def fig_scaling(rows):
    methods = (("Q-table centralisée", "Livraisons Q-table"), ("Glouton (BFS)", "Livraisons glouton"),
               ("PPO centralisé", "Livraisons PPO"))
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    width = 0.26
    x = np.arange(len(ROBOTS))
    for i, (label, col) in enumerate(methods):
        stats = [mean_std(rows[n][col]) for n in ROBOTS]
        means, stds = [m for m, _ in stats], [s for _, s in stats]
        low = [min(m, sd) for m, sd in zip(means, stds)]       # une livraison ne descend pas sous 0
        bars = ax.bar(x + (i - 1) * width, means, width - 0.03, yerr=[low, stds], color=SERIES[i], label=label,
                      error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
        for bar, m in zip(bars, means):
            ax.annotate(f"{m:.1f}", (bar.get_x() + bar.get_width() / 2, m), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [f"{n} robots" for n in ROBOTS])
    ax.set_ylabel("Livraisons par épisode")
    ax.legend(loc="upper left", ncols=3, fontsize=8.5, bbox_to_anchor=(0, 1.13))
    ax.grid(axis="x", visible=False)
    save(fig, "scaling")


def fig_states(rows):
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    width = 0.36
    x = np.arange(len(ROBOTS))
    theo = [float(rows[n]["États théoriques"]) for n in ROBOTS]
    seen = [float(rows[n]["États visités"].replace(" ", "")) for n in ROBOTS]
    for i, (label, vals) in enumerate((("États théoriques", theo), ("États visités à l'entraînement", seen))):
        bars = ax.bar(x + (i - 0.5) * width, vals, width - 0.03, color=SERIES[i], label=label)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.1e}".replace("e+0", "e").replace("e+", "e"),
                        (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(1e4, 1e15)
    ax.set_xticks(x, [f"{n} robots" for n in ROBOTS])
    ax.set_ylabel("Nombre d'états (échelle log)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    save(fig, "states")


def fig_q_curves():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.1))
    for i, n in enumerate(ROBOTS):
        x, y = curve(f"q_{n}", "deliveries", 5000)
        a.plot(x, y, color=SERIES[i], linewidth=1.6, label=f"{n} robots")
        x, y = curve(f"q_{n}", "table_mb", 1)
        b.plot(x, y, color=SERIES[i], linewidth=1.6, label=f"{n} robots")
    b.axhline(600, color=MUTED, linewidth=1, linestyle="--")
    b.annotate("plafond 600 Mo", (0, 600), xytext=(2, 4), textcoords="offset points", fontsize=8, color=MUTED)
    a.set_title("Livraisons (moyenne glissante, 5000 épisodes)", fontsize=9.5, color=INK)
    b.set_title("Taille de la table Q (Mo)", fontsize=9.5, color=INK)
    for ax in (a, b):
        ax.set_xlabel("Pas d'entraînement (millions)")
    a.legend(fontsize=8.5)
    save(fig, "q_curves")


def fig_ppo_curves(rows):
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for i, n in enumerate(ROBOTS):
        x, y = curve(f"ppo_{n}", "deliveries", 500)
        ax.plot(x, y, color=SERIES[i], linewidth=1.6, label=f"PPO, {n} robots")
        g, _ = mean_std(rows[n]["Livraisons glouton"])
        ax.axhline(g, color=SERIES[i], linewidth=1, linestyle=":")
    ax.plot([], [], color=MUTED, linewidth=1, linestyle=":", label="glouton\n(même couleur,\nmême flotte)")
    ax.set_xlabel("Pas d'entraînement (millions)")
    ax.set_ylabel("Livraisons par épisode\n(moyenne glissante, 500 ép.)")
    ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "ppo_curves")


def fig_ladder():
    """Échelle d'information sur big, si les benchmarks ont tourné (results/ladder_*.csv)."""
    entries = []
    for name, label in (("ladder_obs", None), ("ladder_state", "Planner, info=state"),
                        ("ladder_future", "Planner, info=future")):
        path = ROOT / "results" / f"{name}.csv"
        if not path.is_file():
            print("skipped ladder:", path, "missing")
            return
        df = pd.read_csv(path)
        for algo, group in df.groupby("algorithm", sort=False):
            lab = label or {"greedy": "Glouton (BFS)", "planner": "Planner, info=obs"}[algo]
            entries.append((lab, group["deliveries"].mean(), group["deliveries"].std()))
    entries.insert(1, ("PPO (12 h)", 18.3, None))       # chiffre publié (LEARNINGS.md), checkpoint non local
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    y = np.arange(len(entries))[::-1]
    for yi, (lab, m, s) in zip(y, entries):
        color = SERIES[2] if lab.startswith("PPO") else MUTED if lab.startswith("Glouton") else SERIES[0]
        ax.barh(yi, m, 0.62, xerr=s, color=color,
                error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
        ax.annotate(f"{m:.1f}", (m + (s or 0), yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [e[0] for e in entries])
    ax.set_xlabel("Livraisons par épisode (big, 100 labyrinthes inédits)")
    ax.grid(axis="y", visible=False)
    save(fig, "ladder")


def eval_points(name):
    """results/logs/<name>.txt -> (épisodes, livraisons) des lignes « eval @ » (labyrinthes inédits)."""
    text = (ROOT / "results" / "logs" / f"{name}.txt").read_text(encoding="utf-8")
    pts = re.findall(r"eval @ (\d+): deliveries ([\d.]+)", text)
    return np.array([int(e) for e, _ in pts]), np.array([float(d) for _, d in pts])


def ladder_mean(name, algo):
    with open(ROOT / "results" / f"{name}.csv", encoding="utf-8") as f:
        return np.mean([float(r["deliveries"]) for r in csv.DictReader(f) if r["algorithm"] == algo])


def fig_big():
    """big : PPO avec et sans relations de conflit (3 graines), glouton et planificateur en repère."""
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    for i, (prefix, label) in enumerate((("f1", "PPO avec relations de conflit"),
                                         ("long", "PPO sans relations de conflit"))):
        runs = [eval_points(f"{prefix}_s{s}") for s in (10000, 20000, 30000)]
        n = min(len(e) for e, _ in runs)
        x = runs[0][0][:n] * BIG_STEPS_PER_EPISODE / 1e6
        y = np.array([d[:n] for _, d in runs])
        ax.fill_between(x, y.min(0), y.max(0), color=SERIES[i], alpha=0.2, linewidth=0)
        ax.plot(x, y.mean(0), color=SERIES[i], linewidth=1.6, marker="o", markersize=3, label=label)
    for value, label, style in ((ladder_mean("ladder_future", "planner"), "planificateur, futur connu", "--"),
                                (ladder_mean("ladder_obs", "greedy"), "glouton", ":")):
        ax.axhline(value, color=MUTED, linewidth=1, linestyle=style)
        ax.annotate(f"{label} ({value:.1f})".replace(".", ","), (20, value), xytext=(0, 3),
                    textcoords="offset points", ha="right", fontsize=8, color=MUTED)
    ax.set_xlabel("Pas d'entraînement (millions)")
    ax.set_ylabel("Livraisons par épisode")
    ax.set_ylim(8, 23)
    ax.legend(fontsize=8.5, loc="upper right")
    save(fig, "big")


if __name__ == "__main__":
    rows = qtable_rows()
    fig_scaling(rows)
    fig_states(rows)
    fig_q_curves()
    fig_ppo_curves(rows)
    fig_ladder()
    fig_big()
