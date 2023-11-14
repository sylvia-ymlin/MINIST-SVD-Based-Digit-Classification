import os
import numpy as np
from plots import (
    plot_accuracy,
    plot_confusion,
)

LABELS = {
    "none": "No preprocessing",
    "center": "Centering",
    "normalize": "Normalization",
    "center_normalize": "Centering + normalization",
}

DATA_DIR = "data"
NUM_SAMPLES = 400
RANK_LIMIT = 50
K_VALUES = [5, 10, 15, 20, 21, 22, 23, 24, 25, 30, 35, 40]
PREPROCESSINGS = ["none", "center", "normalize", "center_normalize"]


def load_data(data_dir):
    raw_train_path = os.path.join(data_dir, "TrainDigitsRaw.npy")
    raw_test_path = os.path.join(data_dir, "TestDigitsRaw.npy")
    if not (os.path.exists(raw_train_path) and os.path.exists(raw_test_path)):
        raise FileNotFoundError(
            "Raw MNIST arrays not found. Run python src/data_preparation.py first."
        )

    train_digits = np.load(raw_train_path)
    test_digits = np.load(raw_test_path)

    train_labels = np.load(os.path.join(data_dir, "TrainLabels.npy"))
    test_labels = np.load(os.path.join(data_dir, "TestLabels.npy"))
    return train_digits, train_labels, test_digits, test_labels

def normalize_columns(x):
    norms = np.linalg.norm(x, axis=0)
    norms[norms == 0] = 1.0
    return x / norms


def build_bases(train_digits, train_labels, mode, num_samples, rank_limit, data_dir):
    cache_path = os.path.join(
        data_dir,
        f"svd_models_{mode}_n{num_samples}_r{rank_limit}.npz",
    )
    if os.path.exists(cache_path):
        stored = np.load(cache_path)
        return stored["bases"], stored["means"]

    bases = []
    means = []
    for digit in range(10):
        digit_subset = train_digits[:, train_labels == digit][:, :num_samples]
        mean = np.zeros(digit_subset.shape[0], dtype=digit_subset.dtype)
        transformed = digit_subset

        if mode in {"center", "center_normalize"}:
            mean = digit_subset.mean(axis=1)
            transformed = digit_subset - mean[:, np.newaxis]
        if mode in {"normalize", "center_normalize"}:
            transformed = normalize_columns(transformed)

        u, _, _ = np.linalg.svd(transformed, full_matrices=False)
        bases.append(u[:, :rank_limit])
        means.append(mean)

    bases = np.stack(bases, axis=0)
    means = np.stack(means, axis=0)
    np.savez(cache_path, bases=bases, means=means)
    return bases, means


def predict(test_digits, bases, means, mode, k):
    residuals = np.empty((10, test_digits.shape[1]), dtype=np.float32)

    for digit in range(10):
        transformed = test_digits
        if mode in {"center", "center_normalize"}:
            transformed = transformed - means[digit][:, np.newaxis]
        if mode in {"normalize", "center_normalize"}:
            transformed = normalize_columns(transformed)

        uk = bases[digit, :, : min(k, bases.shape[2])]
        coefficients = uk.T @ transformed
        projected_energy = np.sum(coefficients * coefficients, axis=0)
        total_energy = np.sum(transformed * transformed, axis=0)
        residuals[digit] = np.sqrt(np.maximum(total_energy - projected_energy, 0.0))

    return np.argmin(residuals, axis=0).astype(int)


def confusion(true_labels, predicted_labels):
    conf = np.zeros((10, 10))
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        conf[int(true_label), int(predicted_label)] += 1
    return 100.0 * conf / conf.sum(axis=1, keepdims=True)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    train_digits, train_labels, test_digits, test_labels = load_data(DATA_DIR)
    curves = {}
    best_run = None

    for mode in PREPROCESSINGS:
        bases, means = build_bases(
            train_digits,
            train_labels,
            mode,
            NUM_SAMPLES,
            RANK_LIMIT,
            DATA_DIR,
        )

        accuracies = []
        for k in K_VALUES:
            preds = predict(test_digits, bases, means, mode, k)
            acc = float(np.mean(preds == test_labels) * 100)
            accuracies.append(acc)

            if best_run is None or acc > best_run["accuracy"]:
                best_run = {
                    "mode": mode,
                    "k": k,
                    "accuracy": acc,
                    "predictions": preds,
                }

        curves[mode] = accuracies

    conf = confusion(test_labels, best_run["predictions"])
    plot_accuracy(K_VALUES, curves)
    plot_confusion(conf)

    np.save(os.path.join(DATA_DIR, "BestPredictions.npy"), best_run["predictions"])
    print(
        f'best: {LABELS[best_run["mode"]]}, '
        f'k={best_run["k"]}, accuracy={best_run["accuracy"]:.2f}%'
    )


if __name__ == "__main__":
    main()
