
from datasets import load_dataset


def load_truthfulqa():
    return load_dataset("truthful_qa", "generation")


def load_halueval():
    return load_dataset("halueval")


def load_fever():
    return load_dataset("fever")