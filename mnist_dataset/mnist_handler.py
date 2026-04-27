import urllib.request
import gzip
import numpy as np
import os

class MnistDataset:
    def load_mnist():
        base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
        files = {
            "x_train": "train-images-idx3-ubyte.gz",
            "y_train": "train-labels-idx1-ubyte.gz",
            "x_test": "t10k-images-idx3-ubyte.gz",
            "y_test": "t10k-labels-idx1-ubyte.gz",
        }

        os.makedirs("data", exist_ok=True)

        for name, file in files.items():
            path = f"data/{file}"
            if not os.path.exists(path):
                print(f"Downloading {file}...")
                urllib.request.urlretrieve(base_url + file, path)

        def load_images(path):
            with gzip.open(path, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            return data.reshape(-1, 28, 28)

        def load_labels(path):
            with gzip.open(path, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
            return data

        x_train = load_images("data/train-images-idx3-ubyte.gz")
        y_train = load_labels("data/train-labels-idx1-ubyte.gz")
        x_test = load_images("data/t10k-images-idx3-ubyte.gz")
        y_test = load_labels("data/t10k-labels-idx1-ubyte.gz")

        return (x_train, y_train), (x_test, y_test)



