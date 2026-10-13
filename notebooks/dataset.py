# %%
import json
import sparse_circuit
import torch

# %%
model_name = "google/gemma-3-270m"
model, pad_id = load_model(model_name)
# %%
blimp_dataset = "/home/users1/hardy/hardy/project/bella_sparse_circuit/blimp/data"
blimp_data = preprocess_blimp_data(blimp_dataset)
# Filter sentence pairs that it suits filter-circuit format
# %%
count = []
output_data = []
with torch.no_grad():
    for item in blimp_data:
        data = json.loads(item)
        if data[0]["prompt"] == data[1]["prompt"]:
            continue
        clean_full = torch.tensor(model.tokenizer(data[0]["prompt"]).input_ids)
        patch_full = torch.tensor(model.tokenizer(data[1]["prompt"]).input_ids)
        if len(clean_full) != len(patch_full):
            continue
        max_len = max(clean_full.size(0), patch_full.size(0))
        diff_mask = clean_full != patch_full
        first_diff_idx = diff_mask.nonzero(as_tuple=True)[0][0].item()
        if (max_len - first_diff_idx) != 2:
            continue
        count.append(first_diff_idx)
        clean_prefix = clean_full[:first_diff_idx]
        patch_prefix = patch_full[:first_diff_idx]
        clean_answer = clean_full[first_diff_idx : max_len - 1]
        patch_answer = patch_full[first_diff_idx : max_len - 1]
        output = {
            "clean_prefix": "".join(model.to_str_tokens(clean_prefix[1:])),
            "patch_prefix": "".join(model.to_str_tokens(patch_prefix[1:])),
            "clean_answer": "".join(model.to_str_tokens(clean_answer)),
            "patch_answer": "".join(model.to_str_tokens(patch_answer)),
        }
        output_data.append(output)
# %%
output_path = "/home/users1/hardy/hardy/project/bella_sparse_circuit/output_data.jsonl"
with open(output_path, "w") as f:
    for output in output_data:
        f.write(json.dumps(output) + "\n")

    # clean_prefix = model.tokenizer(data["clean_prefix"]).input_ids
    # patch_prefix = model.tokenizer(data["patch_prefix"]).input_ids
    # clean_answer = model.tokenizer(data["clean_answer"]).input_ids
    # patch_answer = model.tokenizer(data["patch_answer"]).input_ids
    # clean_full = model.tokenizer(data["clean_prefix"] + data["clean_answer"]).input_ids
    # patch_full = model.tokenizer(data["patch_prefix"] + data["patch_answer"]).input_ids

    # # strip BOS token from response if necessary
    # if clean_answer[0] == model.tokenizer.bos_token_id:
    #     clean_answer = clean_answer[1:]
    # if patch_answer[0] == model.tokenizer.bos_token_id:
    #     patch_answer = patch_answer[1:]

    # # check that answer is one token
    # if len(clean_answer) != 1 or len(patch_answer) != 1:
    #     continue

    # # check that prefixes are the same length
    # if len(clean_prefix) != len(patch_prefix):
    #     continue

    # # check for tokenization mismatches
    # if clean_prefix + clean_answer != clean_full:
    #     continue
    # if patch_prefix + patch_answer != patch_full:
    #     continue

    # if max_length is not None and len(clean_prefix) > max_length:
    #     continue

    # if use_min_length_only:
    #     # restart collection if we've found a new shortest example
    #     if (length := len(clean_prefix)) < min_length:
    #         examples = [] # restart collection
    #         min_length = length
    #     # skip if too long
    #     elif length > min_length:
    #         continue

    # examples.append(data)

    # if len(examples) >= n_examples:
    #     return examples
# %%
