import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf

from sparse_circuit.experiment import Experiment
from sparse_circuit.dataset import Dataset
from sparse_circuit.models import InterpretableModel
from sparse_circuit import metrics

root = lambda: rootutils.find_root()
OmegaConf.register_new_resolver(name="root", resolver=root)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset_path = cfg["dataset"]["path"]
    metric_fn = getattr(metrics, cfg["patch"]["metric"])
    model_name = cfg["llm_model"]["name"]
    dataset = Dataset(dataset_path=dataset_path)
    model = InterpretableModel(model_name=model_name, metric_fn=metric_fn)
    experiment = Experiment(dataset=dataset, model=model)
    experiment.fit()
    experiment.run()
    experiment.evaluate()
    experiment.report()


if __name__ == "__main__":
    main()
