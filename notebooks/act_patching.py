#%%
import sys
sys.path.append('../src')
import random
import numpy as np
import torch
from models import load_model, get_sae_release_id, compute_log_likelihood, load_sae_model
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

torch.set_grad_enabled(False)
#%%
'''
The activation patching:
- Process sentences using encoder and transformers
- Forward pass
- Replace the activation of the bad one with the clean one
- Forward pass
'''
# init model
model_name = 'google/gemma-3-270m'
model, pad_id = load_model(model_name)
release, sae_ids = get_sae_release_id(model_name)
sae_id = sae_ids[2]
sae, hook_block_name = load_sae_model(model_name, release, sae_id)
#%%
# Preprocessing sentences
good_sent = 'The girls clean the house.'
bad_sent = 'The girls cleans the house.'
token_ids = model.to_tokens([good_sent, bad_sent], prepend_bos=True)
#%%
# get the clean activation
# get the bad activation



logits, cache = model.run_with_cache_with_saes(
    good_token_ids,
    saes=[sae],
    use_error_term=True
)

attention_mask = (good_token_ids != pad_id).long()
batch_avg_sent_loglik = compute_log_likelihood(logits, good_token_ids, attention_mask)
feature_acts = cache[hook_block_name]

#%%

# Forward pass, report loss

# Replace the activation of the bad one with the clean one
# Forward pass, report loss
