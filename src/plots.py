import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_accuracy(k_values, curves):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.set_dpi(150)
    best = None

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4C566A")
    ax.spines["bottom"].set_color("#4C566A")
    ax.grid(True, color="#D8DEE9", alpha=0.4, linewidth=0.5)
    ax.tick_params(colors="#4C566A", labelsize=11)

    for mode, label, color in [
        ("none", "No preprocessing", "#4C566A"),
        ("center", "Centering", "#D08770"),
        ("normalize", "Normalization", "#5E81AC"),
        ("center_normalize", "Centering + normalization", "#88C0D0"),
    ]:
        if mode not in curves:
            continue
        accuracies = curves[mode]
        ax.plot(k_values, accuracies, color=color, linewidth=2, marker="o", markersize=3,
                markerfacecolor="white", markeredgewidth=1.2, markeredgecolor=color,
                label=label, zorder=2)

        i = int(max(range(len(accuracies)), key=accuracies.__getitem__))
        k = k_values[i]
        acc = accuracies[i]
        ax.plot(k, acc, "o", markersize=7, color=color, zorder=3)

        if best is None or acc > best["acc"]:
            best = {"label": label, "k": k, "acc": acc, "color": color}

    ax.annotate(
        f'{best["label"]}: {best["acc"]:.2f}% (k={best["k"]})',
        xy=(best["k"], best["acc"]),
        xytext=(best["k"] + 3, best["acc"] - 3.0),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#666", lw=1),
        color=best["color"],
        fontweight="bold",
    )
    ax.set_xlabel("Subspace rank k")
    ax.set_ylabel("Classification accuracy (%)")
    ax.xaxis.label.set_color("#4C566A")
    ax.yaxis.label.set_color("#4C566A")
    ax.legend(loc="lower right", framealpha=0.95, edgecolor="#4C566A",
              facecolor="#ECEFF4")
    plt.savefig("figures/accuracy_vs_rank.png", bbox_inches="tight", pad_inches=0.15, dpi=150)
    plt.close()


def plot_confusion(conf):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    fig.set_dpi(150)
    im = ax.imshow(conf, cmap="Blues", vmin=0, vmax=100)
    off_diag = [(conf[i, j], i, j) for i in range(10) for j in range(10) if i != j]
    top = sorted(off_diag, reverse=True)[:4]
    marked = {(i, j) for _, i, j in top}

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4C566A")
    ax.spines["bottom"].set_color("#4C566A")
    ax.tick_params(colors="#4C566A", labelsize=11)

    for i in range(10):
        for j in range(10):
            val = conf[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if val > 50 else "#4C566A",
                    fontweight="bold" if i == j else "normal")
            if (i, j) in marked:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="#BF616A", linewidth=2.2))

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.xaxis.label.set_color("#4C566A")
    ax.yaxis.label.set_color("#4C566A")
    fig.colorbar(im, ax=ax, shrink=0.82, label="Rate (%)")
    plt.savefig("figures/confusion_matrix.png", bbox_inches="tight", pad_inches=0.15, dpi=150)
    plt.close()
