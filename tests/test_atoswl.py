import sys

sys.path.append("../")

from beta_rec.datasets.atoswl import AtosWl


if __name__ == "__main__":
    dataset = AtosWl()
    dataset.preprocess()
    interactions = dataset.load_interaction()
    print(interactions.head())

