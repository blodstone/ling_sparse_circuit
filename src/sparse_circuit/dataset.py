import json
import logging
from dataclasses import dataclass
from pathlib import Path

import tqdm
from sae_lens import SAETransformerBridge


@dataclass
class BlimpItem:
    dataset: str
    prompt: str
    UID: str
    pairID: str
    id: str
    linguistics_term: str
    label: str


class Dataset:
    def __init__(self, dataset_path: str, batch_size: int):
        self.dataset_path = dataset_path
        self.examples: list[BlimpItem] = []
        self.preprocessors = []
        self.validator = None
        self.logging = logging.getLogger(__name__)
        self.model: SAETransformerBridge | None = None
        if batch_size % 2 != 0:
            raise ValueError("Batch size has to be divisible by two.")
        self.batch_size = batch_size
        self._read_dataset()
        self._validate()

    def set_model(self, model):
        self.model = model

    def _read_dataset(self):
        blimp_dir = Path(self.dataset_path)
        blimp_files = list(blimp_dir.glob("*.jsonl"))
        if len(blimp_files) == 0:
            self.logging.warning(f"No BLIMP files found in {self.dataset_path}")
            raise ValueError
        for file in tqdm.tqdm(blimp_files):
            with file.open() as f:
                blimp = [json.loads(line) for line in f]
                for row in blimp:
                    self.examples.extend(self._build_from_blimp_row(row))

    def _build_from_blimp_row(self, row: dict) -> list[BlimpItem]:
        sentences = [row["sentence_good"], row["sentence_bad"]]
        results = []
        for idx, sentence in enumerate(sentences):
            prompt = sentence
            blimp_item = BlimpItem(
                "blimp",
                prompt,
                row["UID"],
                row["pairID"],
                str(idx),
                row["linguistics_term"],
                "bad" if idx == 1 else "good",
            )
            results.append(blimp_item)
        return results

    def set_validator(self, validator):
        self.validator = validator

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> BlimpItem:
        return self.examples[idx]

    def __iter__(self):
        indices = list(range(len(self.examples)))
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            samples = [self.examples[idx].prompt for idx in batch_indices]
            if self.model is not None:
                input_ids = self.model.to_tokens(samples)
                yield input_ids
            else:
                raise ValueError("Model is not defined yet.")

    def _validate(self):
        if self.validator is not None:
            self.examples = [
                example for example in self.examples if self.validator(example)
            ]

    def _preprocessors(self):
        for preprocessor in self.preprocessors:
            for i in range(len(self.examples)):
                self.examples[i] = preprocessor(self.examples[i])
