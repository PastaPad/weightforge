"""Weight-space modeling: encode, generate, and edit checkpoint weights."""
from .corpus import (Checkpoint, assert_compatible, discover, load_checkpoint,
                     load_hf_checkpoint)
from .evaluate import (EvalSet, interpolation_curve, load_eval_set, ridge_loo_r2,
                       scalar_statistics, task_vector_curve, weight_features)
from .functional import (apply_vector, evaluate_nll, interpolate, load_model,
                         model_from_state, task_vector)
from .hf_models import Qwen2Config, load_qwen2_state
from .hf_experiment import (HFEvalSet, PairResult, load_pair, run_corpus, run_pair,
                            save_result, summarize_corpus)

__all__ = [
    "Checkpoint", "discover", "load_checkpoint", "load_hf_checkpoint",
    "assert_compatible",
    "EvalSet", "interpolation_curve", "load_eval_set", "ridge_loo_r2",
    "scalar_statistics", "task_vector_curve", "weight_features",
    "apply_vector", "evaluate_nll", "interpolate", "load_model",
    "model_from_state", "task_vector",
    "Qwen2Config", "load_qwen2_state",
    "HFEvalSet", "PairResult", "load_pair", "run_pair", "save_result",
    "run_corpus", "summarize_corpus",
]
