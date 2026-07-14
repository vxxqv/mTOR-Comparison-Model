from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import solve_ivp

plt.style.use("seaborn-v0_8-whitegrid")

OUTPUT = Path(__file__).resolve().parent / "outputs"
OUTPUT.mkdir(exist_ok=True)

TIME = np.linspace(0.0, 240.0, 1201)

PARAMETERS = {
    "m1_basal": 0.010,
    "m1_on": 0.120,
    "m1_off": 0.030,
    "m2_basal": 0.012,
    "m2_on": 0.105,
    "m2_off": 0.030,
    "feedback": 0.400,
    "protein_on": 0.055,
    "protein_off": 0.012,
    "actin_on": 0.085,
    "actin_off": 0.022,
    "ampar_on": 0.100,
    "ampar_off": 0.018,
    "weight_on": 0.052,
    "weight_off": 0.006,
}

CONDITIONS = {
    "Wild type": (1.00, 1.00),
    "Angelman-like": (1.60, 0.65),
    "Lower mTORC1": (1.00, 0.65),
    "Restore mTORC2": (1.60, 1.00),
}

COLORS = {
    "Wild type": "#264653",
    "Angelman-like": "#D1495B",
    "Lower mTORC1": "#F4A261",
    "Restore mTORC2": "#2A9D8F",
}


def hill(x, half=0.40, power=2.0):
    x = np.maximum(x, 0.0)
    return x**power / (half**power + x**power + 1e-12)


def stimulus(t):
    centers = np.array([2.0, 2.35, 2.70, 3.05, 3.40])
    return float(np.exp(-((t - centers) / 0.075) ** 2).sum())


def model(t, state, m1_drive, m2_drive):
    m1, m2, protein, actin, ampar, weight = state
    pulse = hill(stimulus(t), 0.35, 2.0)
    p = PARAMETERS
    dm1 = m1_drive * (p["m1_basal"] + p["m1_on"] * (0.80 * pulse + 0.20 * m2)) * (1.0 - m1) - p["m1_off"] * m1
    dm2 = m2_drive * (p["m2_basal"] + p["m2_on"] * pulse) * (1.0 - m2) - p["m2_off"] * m2 - p["feedback"] * m1 * m2
    dprotein = p["protein_on"] * m1 * (1.0 - protein) - p["protein_off"] * protein
    dactin = p["actin_on"] * m2 * (1.0 - actin) - p["actin_off"] * actin
    dampar = p["ampar_on"] * hill(pulse + 0.55 * protein, 0.50, 2.0) * (1.0 - ampar) - p["ampar_off"] * ampar
    support = protein * actin * ampar
    dweight = p["weight_on"] * support * (1.70 - weight) - p["weight_off"] * (weight - 1.0)
    return [dm1, dm2, dprotein, dactin, dampar, dweight]


def simulate(m1_drive, m2_drive, times=TIME):
    start = [min(0.18 * m1_drive, 0.90), min(0.17 * m2_drive, 0.90), 0.10, 0.14 * m2_drive, 0.10, 1.0]
    solution = solve_ivp(model, (times[0], times[-1]), start, t_eval=times, args=(m1_drive, m2_drive), method="LSODA", rtol=1e-7, atol=1e-9)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution


def measurements(solution):
    return {
        "mTORC1 peak": float(solution.y[0].max()),
        "mTORC2 peak": float(solution.y[1].max()),
        "Protein AUC": float(np.trapezoid(solution.y[2], solution.t)),
        "Actin AUC": float(np.trapezoid(solution.y[3], solution.t)),
        "Final strength": float(solution.y[5, -1]),
    }


def save(fig, filename):
    fig.savefig(OUTPUT / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_figures(solutions, results, grid, landscape):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    series = [(0, "mTORC1 activity"), (1, "mTORC2 activity"), (2, "Protein synthesis"), (5, "Synaptic strength")]
    for ax, (index, label) in zip(axes.flat, series):
        for name, solution in solutions.items():
            ax.plot(solution.t, solution.y[index], color=COLORS[name], linewidth=2.2, label=name)
        ax.set_xlabel("Time after stimulation (min)")
        ax.set_ylabel(label)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("mTOR signalling after a brief simulated stimulus", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save(fig, "01_time_courses.png")

    names = list(CONDITIONS)
    endpoints = [results[name]["Final strength"] for name in names]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.bar(names, endpoints, color=[COLORS[name] for name in names], edgecolor="#222222")
    ax.axhline(results["Wild type"]["Final strength"], color=COLORS["Wild type"], linestyle="--", linewidth=1.6)
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set_ylabel("Synaptic strength at 240 min")
    ax.set_title("Comparison of the four model conditions", fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    save(fig, "02_condition_comparison.png")

    cmap = LinearSegmentedColormap.from_list("balance", ["#17324D", "#2A9D8F", "#F4D35E", "#E76F51"])
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    step = grid[1] - grid[0]
    image = ax.imshow(landscape, origin="lower", extent=[grid.min() - step / 2, grid.max() + step / 2, grid.min() - step / 2, grid.max() + step / 2], aspect="auto", cmap=cmap)
    for name, (m1_drive, m2_drive) in CONDITIONS.items():
        ax.scatter(m1_drive, m2_drive, s=75, color=COLORS[name], edgecolor="white", linewidth=1.2)
        shift = -5 if m1_drive == grid.max() else 5
        alignment = "right" if m1_drive == grid.max() else "left"
        ax.annotate(name, (m1_drive, m2_drive), xytext=(shift, 5), textcoords="offset points", fontsize=8, color="white", ha=alignment)
    fig.colorbar(image, ax=ax, label="Synaptic strength at 240 min")
    ax.set_xlabel("mTORC1 input multiplier")
    ax.set_ylabel("mTORC2 input multiplier")
    ax.set_title("mTORC1–mTORC2 balance map", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save(fig, "03_balance_heatmap.png")

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    for name, solution in solutions.items():
        points = np.linspace(0, solution.t.size - 1, 14, dtype=int)
        ax.plot(solution.y[0], solution.y[1], color=COLORS[name], linewidth=2.2, label=name)
        ax.scatter(solution.y[0, points], solution.y[1, points], color=COLORS[name], s=18)
    ax.set_xlabel("mTORC1 activity")
    ax.set_ylabel("mTORC2 activity")
    ax.set_title("How the two complexes change together", fontsize=16, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "04_mtor_trajectories.png")

    radar_labels = ["mTORC1 peak", "mTORC2 peak", "Protein AUC", "Actin AUC", "Final strength"]
    raw = np.array([[results[name][label] for label in radar_labels] for name in names])
    scaled = raw / np.maximum(raw.max(axis=0), 1e-12)
    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False)
    closed_angles = np.append(angles, angles[0])
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={"projection": "polar"})
    for name, values in zip(names, scaled):
        closed = np.append(values, values[0])
        ax.plot(closed_angles, closed, color=COLORS[name], linewidth=2.2, label=name)
        ax.fill(closed_angles, closed, color=COLORS[name], alpha=0.06)
    ax.set_xticks(angles, [label.replace(" ", "\n") for label in radar_labels], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Normalized model profile", fontsize=16, fontweight="bold", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.12), fontsize=8)
    fig.tight_layout()
    save(fig, "05_radar_profile.png")


def main():
    solutions = {name: simulate(*drives) for name, drives in CONDITIONS.items()}
    results = {name: measurements(solution) for name, solution in solutions.items()}
    grid = np.linspace(0.60, 1.60, 25)
    grid_time = np.linspace(0.0, 240.0, 481)
    landscape = np.zeros((grid.size, grid.size))
    for row, m2_drive in enumerate(grid):
        for column, m1_drive in enumerate(grid):
            landscape[row, column] = simulate(m1_drive, m2_drive, grid_time).y[5, -1]
    create_figures(solutions, results, grid, landscape)
    print("Condition | mTORC1 peak | mTORC2 peak | Protein AUC | Actin AUC | Final strength")
    for name in CONDITIONS:
        values = results[name]
        print(f"{name} | {values['mTORC1 peak']:.4f} | {values['mTORC2 peak']:.4f} | {values['Protein AUC']:.2f} | {values['Actin AUC']:.2f} | {values['Final strength']:.4f}")
    print(f"Grid range | {landscape.min():.4f} to {landscape.max():.4f} | {grid.size * grid.size} simulations")
    print(f"Created 5 figures in {OUTPUT}")


if __name__ == "__main__":
    main()
