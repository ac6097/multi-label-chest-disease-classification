import torch


FEATURE_POOLINGS = {"feature_mean": "mean", "feature_max": "max"}
PREDICTION_POOLINGS = {"logit_mean": "mean", "probability_mean": "mean",
                      "probability_max": "max"}


def pool_by_study(values, study_index, num_studies, mode="mean"):
    if values.ndim != 2 or study_index.ndim != 1 or values.shape[0] != study_index.shape[0]:
        raise ValueError("Expected values [images, features] and a matching study_index")
    if study_index.numel() and (study_index.min() < 0 or study_index.max() >= num_studies):
        raise ValueError("study_index contains an out-of-range Study index")
    if mode == "mean":
        pooled = values.new_zeros((num_studies, values.shape[1]))
        pooled.index_add_(0, study_index, values)
        counts = values.new_zeros((num_studies, 1))
        counts.index_add_(0, study_index, values.new_ones((values.shape[0], 1)))
        if torch.any(counts == 0):
            raise ValueError("Every Study in the batch must have at least one image")
        return pooled / counts
    if mode == "max":
        pooled = values.new_full((num_studies, values.shape[1]), -torch.inf)
        indices = study_index[:, None].expand(-1, values.shape[1])
        pooled.scatter_reduce_(0, indices, values, reduce="amax", include_self=True)
        if torch.isinf(pooled).any():
            raise ValueError("Every Study in the batch must have at least one image")
        return pooled
    raise ValueError(f"Unsupported Study pooling mode: {mode}")
