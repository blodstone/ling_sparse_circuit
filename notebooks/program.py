# %%
from sparse_circuit.dataset import Dataset
import rootutils
from sae_lens import SAETransformerBridge

# %%
path = rootutils.find_root(indicator="pyproject.toml")
dataset = Dataset(
    dataset_path=f"{path}/data/raw/blimp/data",
    batch_size=4,
)
# %%
model = SAETransformerBridge.boot_transformers("gpt2")
# %%
print(model)
# Make a dataset

# Setup the model
# Discover circuits
# %%
