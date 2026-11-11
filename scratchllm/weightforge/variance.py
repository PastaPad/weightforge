"""Decompose a family of weight deltas into recipe and data components.

A controlled family of public derivative models is a two-way design: several
training recipes (LoRA rank 1 on one matrix, LoRA rank 32 everywhere, ...) times
several training datasets. If weight deltas factor cleanly, then

    delta(recipe, data) = mean + recipe_effect + data_effect + residual

and the two effects should be roughly additive. This module answers how much of
the delta norm each factor explains, using only the Gram matrix so that no extra
copies of the deltas are ever materialized.

It also computes a stricter test: are the data-specific *residual* directions the
same across recipes? If they are, a delta's task component is recipe-independent
and can be learned once and reused, which is exactly what a generator needs.
"""
from __future__ import annotations

import numpy as np


def total_sum_of_squares(gram: np.ndarray) -> float:
    return float(np.trace(gram) - gram.sum() / gram.shape[0])


def group_sum_of_squares(gram: np.ndarray, labels: list[str]) -> float:
    """Between-group sum of squares for a partition of the deltas."""
    total = 0.0
    for level in sorted(set(labels)):
        members = [i for i, label in enumerate(labels) if label == level]
        block = gram[np.ix_(members, members)]
        total += block.sum() / len(members)
    return float(total - gram.sum() / gram.shape[0])


def two_way_anova(gram: np.ndarray, recipe: list[str], data: list[str]) -> dict:
    """Variance explained by recipe, data, and their interaction.

    With one model per (recipe, data) cell the interaction and the residual are
    not separable; they are reported together as ``interaction``.
    """
    total = total_sum_of_squares(gram)
    recipe_ss = group_sum_of_squares(gram, recipe)
    data_ss = group_sum_of_squares(gram, data)
    cells = [f"{r}|{d}" for r, d in zip(recipe, data)]
    cell_ss = group_sum_of_squares(gram, cells)
    interaction = max(cell_ss - recipe_ss - data_ss, 0.0)
    share = lambda value: value / total if total else float("nan")
    return {
        "total": total,
        "recipe": recipe_ss,
        "data": data_ss,
        "interaction": interaction,
        "recipe_share": share(recipe_ss),
        "data_share": share(data_ss),
        "interaction_share": share(interaction),
    }


def leave_one_out_from_gram(gram: np.ndarray, rank: int | None = None) -> list[dict]:
    """Delta-space fidelity of reconstructing each delta from all the others.

    Pure linear algebra on the Gram matrix: no model loading and no scoring, so
    this is the cheap way to ask whether a basis spans a held-out delta. The NLL
    of the reconstructed model answers a different question and needs a forward
    pass.
    """
    results = []
    for index in range(gram.shape[0]):
        others = [j for j in range(gram.shape[0]) if j != index]
        sub_gram = gram[np.ix_(others, others)]
        sub_inner = gram[index][others]
        eigenvalues, eigenvectors = np.linalg.eigh(sub_gram)
        keep = eigenvalues > max(eigenvalues.max() * 1e-10, 0.0)
        if rank is not None:
            order = np.argsort(eigenvalues)[::-1][:rank]
            keep = np.zeros(len(eigenvalues), dtype=bool)
            keep[order] = True
        values = np.where(keep, eigenvalues, 0.0)
        coefficients = eigenvectors @ (np.where(keep, eigenvectors.T @ sub_inner, 0.0) /
                                       (values + 1e-30))
        full = np.zeros(gram.shape[0])
        for position, j in enumerate(others):
            full[j] = coefficients[position]
        quadratic = float(full @ gram @ full)
        covariance = float(full @ gram[:, index])
        true_norm = float(gram[index, index] ** 0.5)
        # A zero-length reconstruction has no direction; report 0 rather than NaN
        # so the "orthogonal basis" case reads as "nothing was recovered".
        results.append({
            "index": index,
            "cosine": covariance / (quadratic ** 0.5 * true_norm)
            if quadratic > 0 and true_norm else 0.0,
            "norm_ratio": quadratic ** 0.5 / true_norm if true_norm else 0.0,
            "rank": rank,
        })
    return results


def residual_cosines(gram: np.ndarray, recipe: list[str], data: list[str]) -> dict:
    """Cosine between the data-specific residuals of each dataset, across recipes.

    Every quantity is a linear combination of Gram entries, so this works from
    the Gram matrix alone.
    """
    names = list(dict.fromkeys(recipe))
    levels = list(dict.fromkeys(data))
    indices = {(r, d): i for i, (r, d) in enumerate(zip(recipe, data))}
    scale = {level: 1.0 / sum(1 for r in recipe if r == level) for level in names}

    def dot(vector: dict[int, float], other: dict[int, float]) -> float:
        return sum(weight * other.get(index, 0.0) for index, weight in vector.items())

    def mean_vector(name: str) -> dict[int, float]:
        return {indices[(name, d)]: scale[name] for d in levels if (name, d) in indices}

    results = {}
    for level in levels:
        terms = []
        for name in names:
            terms.append((name, mean_vector(name)))
        left = {indices[(names[0], level)]: 1.0}
        right = {indices[(names[1], level)]: 1.0}
        left_residual = left.copy()
        right_residual = right.copy()
        for name, mean in terms:
            for index, weight in mean.items():
                left_residual[index] = left_residual.get(index, 0.0) - weight
                right_residual[index] = right_residual.get(index, 0.0) - weight
        numerator = dot(left_residual, right_residual)
        left_norm = dot(left_residual, left_residual) ** 0.5
        right_norm = dot(right_residual, right_residual) ** 0.5
        results[level] = numerator / (left_norm * right_norm) if left_norm and right_norm else float("nan")
    return results
