import matplotlib.pyplot as plt
import numpy as np
import os

def plot_fairness_results(metrics_dict, title, filename):

    os.makedirs("plots", exist_ok=True)
    full_path = os.path.join("plots", filename)

    labels = []
    values = []

    for k, v in metrics_dict.items():
        if isinstance(v, (list, tuple)):
            for i, val in enumerate(v):
                labels.append(f"{k} (group {i})")
                values.append(val)
        elif hasattr(v, 'values') and hasattr(v, 'index'):  # pandas Series
            for subk, subv in zip(v.index, v.values):
                labels.append(f"{k} ({subk})")
                values.append(subv)
        elif isinstance(v, dict):
            for subk, subv in v.items():
                labels.append(f"{k} ({subk})")
                values.append(subv)
        elif isinstance(v, (int, float, np.float64)):
            labels.append(k)
            values.append(v)
        else:
            print(f"[WARN] Skipped non-numeric value: {k} -> {v}")

    # Filter for numerical values
    labels_values = [(l, v) for l, v in zip(labels, values) if isinstance(v, (int, float, np.float64))]
    if not labels_values:
        print("[WARN] Nessun valore numerico da plottare.")
        return

    labels, values = zip(*labels_values)
    sorted_indices = np.argsort(values)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]

    # Generating colours
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(sorted_labels))]

    plt.figure(figsize=(12, max(6, len(sorted_labels) * 0.4)))
    bars = plt.barh(sorted_labels, sorted_values, color=colors)
    plt.xlabel("Value")
    plt.title(title)
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)

    # Adding decimal values
    for bar, value in zip(bars, sorted_values):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{value:.3f}", va='center', ha='left', fontsize=9)

    plt.tight_layout()
    plt.savefig(full_path)
    plt.close()
    print(f"Fairness plot saved as: {filename}")
