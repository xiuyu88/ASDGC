"""Evaluation and normalization utilities used by ASDGC."""

import numpy as np


def metric(pred, real):
    """Compute RSE, RAE, and empirical correlation."""
    pred = pred.reshape(-1, pred.shape[-1])
    real = real.reshape(-1, real.shape[-1])
    eps = np.finfo(np.float64).eps

    rse_denominator = np.sqrt(np.sum((real - real.mean()) ** 2))
    rse = np.sqrt(np.sum((pred - real) ** 2)) / max(rse_denominator, eps)

    rae_denominator = np.sum(np.abs(real - real.mean()))
    rae = np.sum(np.abs(pred - real)) / max(rae_denominator, eps)


    sigma_p = pred.std(axis=0)
    sigma_g = real.std(axis=0)
    mean_p = pred.mean(axis=0)
    mean_g = real.mean(axis=0)
    valid = (sigma_p > eps) & (sigma_g > eps)

    if np.any(valid):
        correlation = (
            ((pred - mean_p) * (real - mean_g)).mean(axis=0)
            / np.maximum(sigma_p * sigma_g, eps)
        )[valid].mean()
    else:
        correlation = 0.0

    return {
        "RSE": float(rse),
        "RAE": float(rae),
        "CORR": float(correlation),
    }


def normalize_data(data, fit_end_idx=None):
    """Fit zero-mean normalization on the training range and transform all data."""
    fit_data = data if fit_end_idx is None else data[:fit_end_idx]
    mean = fit_data.mean(axis=0)
    std = fit_data.std(axis=0) + 1e-8
    return (data - mean) / std, mean, std
