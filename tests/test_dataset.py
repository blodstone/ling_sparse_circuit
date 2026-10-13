import pytest
import rootutils
from sparse_circuit.dataset import Dataset
from sae_lens import SAETransformerBridge


class TestDataset:
    @pytest.fixture(scope="session")
    def shared_dataset(self):
        path = rootutils.find_root(search_from=__file__, indicator="pyproject.toml")
        dataset = Dataset(
            dataset_path=f"{path}/data/raw/blimp/data",
            batch_size=4,
        )
        model_name = "gpt2"
        model = SAETransformerBridge.boot_transformers(model_name, device="cuda")
        dataset.set_model(model)
        return dataset

    def test_access(self, shared_dataset):
        assert len(shared_dataset) != 0  # simple check if the creation succeed or not

    def test_index(self, shared_dataset):
        first_item_good = shared_dataset[0]
        assert type(first_item_good.dataset) == str
        assert type(first_item_good.prompt) == str
        assert type(first_item_good.UID) == str
        assert type(first_item_good.pairID) == str
        assert type(first_item_good.id) == str
        assert type(first_item_good.linguistics_term) == str
        assert type(first_item_good.label) == str

    def test_iter(self, shared_dataset):
        items = [sample for sample in shared_dataset]
        assert items[0].shape[0] == 4
