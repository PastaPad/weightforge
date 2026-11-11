"""Command-line interface for weight-space modeling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build import build_corpus
from .corpus import load_checkpoint
from .evaluate import (interpolation_curve, load_eval_set, pearson, ridge_loo_r2,
                       scalar_statistics, task_vector_curve, weight_features)
from .functional import interpolate
from .hf_experiment import (HFEvalSet, load_pair, run_corpus, run_pair, save_result,
                            summarize_corpus)


def _eval_sets(args):
    return [
        load_eval_set("base", args.base_data, context=args.context, seed=args.seed,
                      batch_size=args.batch_size, eval_batches=args.eval_batches),
        load_eval_set("dialogue", args.dialogue_data, context=args.context, dialogue=True,
                      seed=args.seed, batch_size=args.batch_size,
                      eval_batches=args.eval_batches,
                      max_windows=args.batch_size * args.eval_batches),
    ]


def _save(results, out):
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"saved {path}")


def smoke(args):
    base = load_checkpoint(args.base)
    dialogue = load_checkpoint(args.dialogue)
    eval_sets = _eval_sets(args)
    alphas = [round(0.1 * index, 1) for index in range(11)]
    scales = [round(0.25 * index, 2) for index in range(7)]

    print("WEIGHTFORGE SMOKE TEST")
    print(f"base     {base.path} ({base.num_parameters:,} params, step {base.step}, "
          f"val NLL {base.best_val:.4f})")
    print(f"dialogue {dialogue.path} ({dialogue.num_parameters:,} params, step {dialogue.step}, "
          f"val NLL {dialogue.best_val:.4f})")
    print()

    curve = interpolation_curve(base, dialogue, alphas, eval_sets, args.device, args.batch_size)
    print("INTERPOLATION (linear weight mix)")
    print(f"{'alpha':>6} {'base-NLL':>10} {'dialogue-NLL':>13}")
    for row in curve:
        print(f"{row['alpha']:>6.1f} {row['base']:>10.4f} {row['dialogue']:>13.4f}")
    print()

    vector_curve = task_vector_curve(base, dialogue, scales, eval_sets, args.device, args.batch_size)
    print("TASK VECTOR (base + scale * (dialogue - base))")
    print(f"{'scale':>6} {'base-NLL':>10} {'dialogue-NLL':>13}")
    for row in vector_curve:
        print(f"{row['scale']:>6.2f} {row['base']:>10.4f} {row['dialogue']:>13.4f}")
    print()

    features = [weight_features(interpolate(base.state_dict, dialogue.state_dict, alpha))
                for alpha in alphas]
    stats = scalar_statistics(base, dialogue, alphas)
    targets = {name: [row[name] for row in curve] for name in ("base", "dialogue")}

    print("PROBE (weight features -> NLL)")
    print("  scalar correlations along the interpolation path:")
    for stat, values in stats.items():
        for name, target in targets.items():
            print(f"    {stat:<22} vs {name}-NLL: r = {pearson(values, target):+.4f}")
    print("  leave-one-out ridge R^2:")
    for name, values in targets.items():
        r2, _ = ridge_loo_r2(features, values, l2=args.probe_l2)
        print(f"    {name}-NLL: R^2 = {r2:.4f}")
    print()

    _save(dict(base=str(base.path), dialogue=str(dialogue.path),
               interpolation=curve, task_vector=vector_curve,
               scalar_statistics=stats,
               probe={name: ridge_loo_r2(features, values, l2=args.probe_l2)[0]
                      for name, values in targets.items()}), args.out)


def hf_smoke(args):
    base, tuned = load_pair(args.base_id, args.tuned_id)
    eval_set = HFEvalSet.from_file(args.val, window=args.window, count=args.windows,
                                   seed=args.seed)

    print("WEIGHTFORGE HF SMOKE TEST")
    print(f"base   {args.base_id} ({base.num_parameters:,} params)")
    print(f"tuned  {args.tuned_id} ({tuned.num_parameters:,} params)")
    print(f"eval   {args.val} ({len(eval_set.windows)} x {args.window} tokens)")
    print()

    result = run_pair(base, tuned, eval_set, device=args.device,
                      batch_size=args.batch_size, probe_l2=args.probe_l2,
                      progress=lambda *_: None)
    print()
    print("INTERPOLATION (linear weight mix)")
    print(f"{'alpha':>6} {'NLL':>10}")
    for alpha, nll in zip(result.alphas, result.interpolation[eval_set.name]):
        print(f"{alpha:>6.2f} {nll:>10.4f}")
    print()
    print("TASK VECTOR (base + scale * (tuned - base))")
    print(f"{'scale':>6} {'NLL':>10}")
    for scale, nll in zip(result.scales, result.task_vector[eval_set.name]):
        print(f"{scale:>6.2f} {nll:>10.4f}")
    print()
    print("PROBE")
    print(f"  pearson(alpha, NLL): {result.probe['pearson_alpha_nll'][eval_set.name]:+.4f}")
    print("  ridge LOO R^2: "
          f"{result.probe['ridge_loo_r2_interpolation_nll'][eval_set.name]:.4f}")
    print(f"  connectivity: {result.connectivity['label']} "
          f"(median cosine {result.connectivity['median_cosine']:.3f})")

    if args.out:
        print(f"saved {save_result(result, args.out)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scratchllm.weightforge",
                                     description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_eval(p):
        p.add_argument("--base", required=True, help="base checkpoint (best.pt)")
        p.add_argument("--dialogue", required=True, help="dialogue/task checkpoint (best.pt)")
        p.add_argument("--base-data", required=True, help="prepared base validation dataset")
        p.add_argument("--dialogue-data", required=True,
                       help="prepared dialogue validation dataset")
        p.add_argument("--context", type=int, default=512)
        p.add_argument("--device", default="cuda")
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--eval-batches", type=int, default=8)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", default=None, help="write results JSON to this path")

    p_smoke = sub.add_parser("smoke", help="interpolation, task-vector, and probe smoke test")
    add_eval(p_smoke)
    p_smoke.add_argument("--probe-l2", type=float, default=10.0)
    p_smoke.set_defaults(func=smoke)

    p_build = sub.add_parser("build-corpus",
                             help="prepare synthetic tasks and train a small model corpus")
    p_build.add_argument("--config", required=True, help="corpus configuration JSON")
    p_build.set_defaults(func=lambda args: print(
        f"corpus: {len(build_corpus(args.config))} models"))

    p_hf = sub.add_parser("hf-smoke",
                          help="interpolation/task-vector/probe sweep on two HF checkpoints")
    p_hf.add_argument("--base-id", required=True, help="e.g. Qwen/Qwen2.5-0.5B")
    p_hf.add_argument("--tuned-id", required=True, help="e.g. Qwen/Qwen2.5-0.5B-Instruct")
    p_hf.add_argument("--val", required=True, help="prepared uint32 validation token stream")
    p_hf.add_argument("--window", type=int, default=1024)
    p_hf.add_argument("--windows", type=int, default=4)
    p_hf.add_argument("--seed", type=int, default=0)
    p_hf.add_argument("--device", default="cpu")
    p_hf.add_argument("--batch-size", type=int, default=1)
    p_hf.add_argument("--probe-l2", type=float, default=10.0)
    p_hf.add_argument("--out", default=None, help="write results JSON to this path")
    p_hf.set_defaults(func=hf_smoke)

    p_corpus = sub.add_parser("hf-corpus",
                              help="run an HF checkpoint-pair corpus spec")
    p_corpus.add_argument("--config", required=True, help="corpus spec JSON")
    p_corpus.add_argument("--out-dir", required=True, help="directory for per-pair results")
    p_corpus.add_argument("--device", default=None, help="override the spec's device")
    p_corpus.add_argument("--force", action="store_true", help="recompute existing pairs")
    p_corpus.set_defaults(func=hf_corpus)

    p_sum = sub.add_parser("hf-summary",
                           help="tabulate connectivity and barriers for a corpus directory")
    p_sum.add_argument("--dir", required=True, help="directory of per-pair result JSONs")
    p_sum.set_defaults(func=hf_summary)

    return parser


def hf_summary(args):
    rows = summarize_corpus(args.dir)
    print(f"{'pair':28s} {'label':22s} {'cos':>6} {'rel_d':>7} {'barrier':>8}")
    for row in rows:
        barrier = max((value["peak_gap"] for value in row["evals"].values()), default=0.0)
        print(f"{row['id']:28s} {row['label']:22s} {row['median_cosine']:>6.3f} "
              f"{row['median_relative_delta']:>7.3f} {barrier:>+8.3f}")
    print(f"{len(rows)} pairs")


def hf_corpus(args):
    results = run_corpus(args.config, args.out_dir, device=args.device,
                         skip_existing=not args.force)
    print(f"pairs: {len(results)}")


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
