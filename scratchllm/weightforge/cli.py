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
from .compose import run_compose, save_compose
from .lowrank import (run_gram, run_lowrank, run_variance, save_lowrank,
                      save_variance)
from .hf_experiment import (HFEvalSet, load_eval_specs, load_pair, run_corpus, run_pair,
                            save_result, summarize_corpus)


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

    p_comp = sub.add_parser("hf-compose",
                            help="test whether several weight deltas compose additively")
    p_comp.add_argument("--base-id", required=True)
    p_comp.add_argument("--tuned-id", action="append", required=True,
                        help="repeatable; each becomes one delta from the base")
    p_comp.add_argument("--eval", action="append", required=True, metavar="NAME=PATH",
                        help="repeatable, e.g. english=/path/val.bin")
    p_comp.add_argument("--window", type=int, default=1024)
    p_comp.add_argument("--windows", type=int, default=4)
    p_comp.add_argument("--device", default="cpu")
    p_comp.add_argument("--batch-size", type=int, default=1)
    p_comp.add_argument("--out", default=None)
    p_comp.set_defaults(func=hf_compose)

    p_lr = sub.add_parser("hf-lowrank",
                          help="reconstruct held-out deltas from a basis of other deltas")
    p_lr.add_argument("--base-id", required=True)
    p_lr.add_argument("--tuned-id", action="append", required=True,
                      help="repeatable; each becomes one delta from the base")
    p_lr.add_argument("--eval", action="append", required=True, metavar="NAME=PATH")
    p_lr.add_argument("--window", type=int, default=1024)
    p_lr.add_argument("--windows", type=int, default=4)
    p_lr.add_argument("--ranks", default="full,2,1",
                      help="comma-separated ranks; 'full' means no truncation")
    p_lr.add_argument("--device", default="cpu")
    p_lr.add_argument("--batch-size", type=int, default=1)
    p_lr.add_argument("--out", default=None)
    p_lr.set_defaults(func=hf_lowrank)

    p_gram = sub.add_parser("hf-gram",
                            help="cosine structure of a set of deltas (no forward pass)")
    p_gram.add_argument("--base-id", required=True)
    p_gram.add_argument("--model", action="append", required=True)
    p_gram.add_argument("--out", default=None)
    p_gram.set_defaults(func=hf_gram)

    p_var = sub.add_parser("hf-variance",
                           help="recipe x data variance decomposition of a delta family")
    p_var.add_argument("--base-id", required=True)
    p_var.add_argument("--cell", action="append", required=True, metavar="RECIPE,DATA,MODEL_ID",
                       help="repeatable; one model per (recipe, data) cell")
    p_var.add_argument("--eval", action="append", default=None, metavar="NAME=PATH")
    p_var.add_argument("--window", type=int, default=1024)
    p_var.add_argument("--windows", type=int, default=4)
    p_var.add_argument("--device", default="cpu")
    p_var.add_argument("--batch-size", type=int, default=1)
    p_var.add_argument("--out", default=None)
    p_var.set_defaults(func=hf_variance)

    p_sum = sub.add_parser("hf-summary",
                           help="tabulate connectivity and barriers for a corpus directory")
    p_sum.add_argument("--dir", required=True, help="directory of per-pair result JSONs")
    p_sum.set_defaults(func=hf_summary)

    return parser


def _eval_args(args) -> dict:
    specs = []
    for item in args.eval:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--eval must be NAME=PATH, got {item!r}")
        specs.append({"name": name, "val": path, "window": args.window,
                      "windows": args.windows})
    return load_eval_specs(specs)


def hf_compose(args):
    eval_sets = _eval_args(args)
    result = run_compose(args.base_id, args.tuned_id, eval_sets, device=args.device,
                         batch_size=args.batch_size)
    print()
    print(f"{'state':30s} " + " ".join(f"{name:>10s}" for name in eval_sets))
    print(f"{'base':30s} " + " ".join(f"{v:>10.4f}" for v in result.base_nll.values()))
    for name, rows in result.single_nll.items():
        print(f"{name:30s} " + " ".join(f"{v:>10.4f}" for v in rows.values()))
    print(f"{'combined':30s} " + " ".join(f"{v:>10.4f}" for v in result.combined_nll.values()))
    print()
    for name, value in result.interaction.items():
        print(f"  interaction ({name}): {value:+.4f}")
    for name, value in result.delta_cosine.items():
        print(f"  delta cosine {name}: {value:+.3f}")
    if args.out:
        print(f"saved {save_compose(result, args.out)}")


def hf_lowrank(args):
    eval_sets = _eval_args(args)
    ranks = [None if item == "full" else int(item) for item in args.ranks.split(",")]
    result = run_lowrank(args.base_id, args.tuned_id, eval_sets, device=args.device,
                         batch_size=args.batch_size, ranks=ranks)
    print()
    for entry in result.holds:
        print(f"held out {entry['held_out']}")
        for name, rows in entry["ranks"].items():
            scores = " ".join(f"{k}={v:.4f}" for k, v in rows["nll"].items())
            print(f"  {name:6s} {scores} norm_ratio={rows['norm_ratio']:.3f}")
        true_scores = " ".join(f"{k}={v:.4f}" for k, v in entry["true_nll"].items())
        print(f"  true   {true_scores}")
    if args.out:
        print(f"saved {save_lowrank(result, args.out)}")


def hf_gram(args):
    result = run_gram(args.base_id, args.model)
    short = []
    for model in result["models"]:
        label = model.split("/")[-1][:22]
        while label in short:
            label = label[:-1] + "#"
        short.append(label)
    width = max(12, max(len(label) for label in short)) if short else 12
    print()
    print("norms: " + " ".join(f"{name}={value:.3f}"
                               for name, value in zip(short, result["norms"])))
    print("cosines")
    print(" " * (width + 2) + " ".join(f"{name:>{width}s}" for name in short))
    for name, row in zip(short, result["cosines"]):
        print(f"{name:{width}s}  " + " ".join(f"{value:>{width}.3f}" for value in row))
    print()
    if result["duplicates"]:
        print()
        print("WARNING: identical deltas (duplicate uploads?)")
        for left, right in result["duplicates"]:
            print(f"  {left} == {right}")
    print()
    print("leave-one-out reconstruction")
    for name, row in zip(result["models"], result["leave_one_out"]):
        print(f"  {name.split('/')[-1]:44s} cos={row['cosine']:+.3f} "
              f"norm_ratio={row['norm_ratio']:.3f}")
    if args.out:
        print(f"saved {save_variance(result, args.out)}")


def hf_variance(args):
    cells = []
    for item in args.cell:
        parts = item.split(",", 2)
        if len(parts) != 3:
            raise SystemExit(f"--cell must be RECIPE,DATA,MODEL_ID, got {item!r}")
        cells.append(tuple(parts))
    eval_sets = _eval_args(args) if args.eval else None
    result = run_variance(args.base_id, cells, device=args.device, eval_sets=eval_sets,
                          batch_size=args.batch_size)
    print()
    anova = result["anova"]
    print(f"variance decomposition (total {anova['total']:.4g})")
    for name in ("recipe", "data", "interaction"):
        print(f"  {name:12s} {anova[name]:12.4g}  {anova[name + '_share'] * 100:6.2f}%")
    normalized = result["anova_normalized"]
    print("scale-normalized (directional) shares")
    for name in ("recipe", "data", "interaction"):
        print(f"  {name:12s} {'':12s}  {normalized[name + '_share'] * 100:6.2f}%")
    print()
    print("leave-one-out reconstruction from the other deltas")
    for cell, row in zip(result["cells"], result["leave_one_out"]):
        print(f"  {cell['recipe'] + '/' + cell['data']:24s} cos={row['cosine']:+.3f} "
              f"norm_ratio={row['norm_ratio']:.3f}")
    print()
    print("residual (data-specific) cosine between recipes")
    for name, value in result["residual_cosines"].items():
        print(f"  {name:12s} {value:+.3f}")
    overlap = result.get("overlap")
    if overlap:
        print()
        print(f"shared support: {len(overlap['tensors'])} tensor(s)")
        anova = overlap["anova"]
        print(f"  recipe {anova['recipe_share'] * 100:5.2f}%  "
              f"data {anova['data_share'] * 100:5.2f}%  "
              f"interaction {anova['interaction_share'] * 100:5.2f}%")
        for name, value in overlap["residual_cosines"].items():
            print(f"  residual cosine {name:12s} {value:+.3f}")
    if eval_sets:
        print()
        header = "  ".join(f"{name:>9s}" for name in eval_sets)
        print(f"{'cell':22s} {header}")
        for cell in result["cells"]:
            scores = result["nll"].get(cell["model"], {})
            print(f"{cell['recipe'] + '/' + cell['data']:22s} " +
                  "  ".join(f"{scores.get(name, float('nan')):9.4f}" for name in eval_sets))
    if args.out:
        print(f"saved {save_variance(result, args.out)}")


def hf_summary(args):
    rows = summarize_corpus(args.dir)
    print(f"{'pair':32s} {'label':22s} {'cos':>6} {'rel_d':>7} {'changed':>8} {'barrier':>8}")
    for row in rows:
        barrier = max((value["peak_gap"] for value in row["evals"].values()), default=0.0)
        total = row.get("total_relative_delta")
        changed = row.get("changed_fraction")
        print(f"{row['id']:32s} {row['label']:22s} {row['median_cosine']:>6.3f} "
              f"{total if total is not None else float('nan'):>7.3f} "
              f"{100 * changed if changed is not None else float('nan'):>7.1f}% "
              f"{barrier:>+8.3f}")
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
