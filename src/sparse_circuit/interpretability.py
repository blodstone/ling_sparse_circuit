from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch
from sparse_circuit.models import InterpretableModel
from sparse_circuit.numeric import ShiftDataVariance
from sparse_circuit.sparse_activation import SparseAct


@dataclass
class PatchedFeature:
    model_name: str
    sae_release: str
    sae_id: str
    feature_id: int
    n_samples: int
    score: ShiftDataVariance = field(default_factory=ShiftDataVariance)
    activation_clean: ShiftDataVariance = field(default_factory=ShiftDataVariance)
    activation_patched: ShiftDataVariance = field(default_factory=ShiftDataVariance)
    example_indices: list[int] = field(default_factory=list)

    def __init__(self, model_name: str, sae_release: str, sae_id: str):
        self.model_name = model_name
        self.sae_release = sae_release
        self.sae_id = sae_id
        self.n_samples = 0
        
    def add_data(
        self,
        new_score: float,
        new_activation_clean: float,
        new_activation_patched: float,
        example_idx: int,
    ):
        self.n_samples += 1
        self.example_indices.append(example_idx)
        self.score.add_variable(new_score)
        self.activation_clean.add_variable(new_activation_clean)
        self.activation_patched.add_variable(new_activation_patched)
    


class Patcher(ABC):
    @abstractmethod
    def patch(self, batch: int, model: InterpretableModel, **kwargs):
        pass


class AttributePatcher(Patcher):
    def __init__(self, metric_fn):
        self.metric_fn = metric_fn

    def patch(self, batch: int, model: InterpretableModel, **kwargs):
        hook_name = kwargs["hooked_name"]
        sae = kwargs["sae"]
        _, cache = model.run_with_cache(batch)
        x = cache[hook_name]
        f = sae.encode(x)
        x_hat = sae.decode(f)
        res = x_hat - x
        sparse_act = SparseAct(f, res)
        res.grad = torch.zeros_like(res
