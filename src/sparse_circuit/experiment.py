import logging

import torch

from sparse_circuit.dataset import Dataset
from sparse_circuit.models import InterpretableModel


class Experiment:
    def __init__(self, dataset: Dataset, model: InterpretableModel):
        self.dataset = dataset
        self.model = model
        self.logging = logging.getLogger(__name__)

    def setup(self):
        self.setup_dataset()

    def setup_dataset(self):
        self.dataset.set_model(self.model)

    def fit(self):
        self.logging.info("Fit stage")

    def run(self):
        self.logging.info("Run stage")
        with torch.no_grad():
            for samples in self.dataset:
                feature = self.model.patch(samples)

    def evaluate(self):
        self.logging.info("Evaluate stage")

    def report(self):
        self.logging.info("Reporting stage")
