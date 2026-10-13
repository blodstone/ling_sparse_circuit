import torch


# TODO: write a test case for this method
def pair_logit_metric(output_logit, clean_pred, patched_pred):
    """The metric that takes the difference between clean and patched logits
    Args:
        output_logit: a tensor of size Bx1xD which contains the distribution probability of the next token
        clean_pred: the next token prediction of clean activation, a tensor of size B
        patched_pred: the next token prediction of patched activation, a tensor of size B
    Returns:
        the difference of logits
    """
    clean_logit = torch.gather(output_logit, -1, clean_pred.view(1, -1))
    patched_logit = torch.gather(output_logit, -1, patched_pred.view(1, -1))
    diff_logit = patched_logit - clean_logit
    return diff_logit
