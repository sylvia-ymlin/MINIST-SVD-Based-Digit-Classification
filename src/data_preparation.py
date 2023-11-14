import numpy as np
import os
from sklearn.datasets import fetch_openml # use to download MNIST dataset

DATA_DIR = "data"


def download_mnist(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto', data_home=data_dir)
    
    # The MNIST dataset contains 70,000 samples. The first 60,000 are typically used for training and the remaining 10,000 for testing.
    X_train = mnist.data[:60000]
    X_test = mnist.data[60000:]
    y_train = mnist.target[:60000]
    y_test = mnist.target[60000:]
    
    X_train_t = X_train.T.astype(np.float32)
    X_test_t = X_test.T.astype(np.float32)
    
    y_train = y_train.astype(np.int32)
    y_test  = y_test.astype(np.int32)

    train_path = os.path.join(data_dir, "TrainDigitsRaw.npy")
    train_label_path = os.path.join(data_dir, "TrainLabels.npy")
    test_path = os.path.join(data_dir, "TestDigitsRaw.npy")
    test_label_path = os.path.join(data_dir, "TestLabels.npy")

    np.save(train_path, X_train_t)
    np.save(train_label_path, y_train)
    np.save(test_path, X_test_t)
    np.save(test_label_path, y_test)

    print(f"Saved to {data_dir}")
    print(f"Train: Digits {X_train_t.shape}, Labels {y_train.shape}")
    print(f"Test:  Digits {X_test_t.shape}, Labels {y_test.shape}")

if __name__ == "__main__":
    download_mnist(DATA_DIR)
