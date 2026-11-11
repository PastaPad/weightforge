"""Synthetic task generators for the weightforge corpus.

Each generator writes one example per line (or a multi-line record without blank
lines) so ``scratchllm.prepare`` can tokenize it into a base training dataset.
The tasks are deliberately small, controllable, and diverse: they give the
generator distinct capabilities to learn without any external downloads.
"""
from __future__ import annotations

import json
import random
import string
from pathlib import Path


def _write(path, lines) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def arithmetic(path, count: int = 20000, seed: int = 1234) -> None:
    rng = random.Random(seed)
    lines = []
    for _ in range(count):
        a, b = rng.randint(1, 999), rng.randint(1, 999)
        op = rng.choice(["+", "-", "*"])
        result = a + b if op == "+" else a - b if op == "-" else a * b
        lines.append(f"What is {a} {op} {b}? {result}")
    _write(path, lines)


def copy_task(path, count: int = 20000, seed: int = 1234) -> None:
    rng = random.Random(seed)
    lines = []
    for _ in range(count):
        words = ["".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 6)))
                 for _ in range(rng.randint(3, 12))]
        lines.append(f"Repeat: {' '.join(words)}")
    _write(path, lines)


def reverse_task(path, count: int = 20000, seed: int = 1234) -> None:
    rng = random.Random(seed)
    lines = []
    for _ in range(count):
        words = ["".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 6)))
                 for _ in range(rng.randint(3, 12))]
        lines.append(f"Reverse: {' '.join(words)} = {' '.join(reversed(words))}")
    _write(path, lines)


def json_task(path, count: int = 20000, seed: int = 1234) -> None:
    rng = random.Random(seed)
    lines = []
    for _ in range(count):
        obj = {"name": "".join(rng.choice(string.ascii_lowercase) for _ in range(6)),
               "age": rng.randint(1, 99),
               "tags": [rng.choice(["a", "b", "c", "d"]) for _ in range(rng.randint(1, 3))],
               "active": rng.choice([True, False])}
        lines.append(json.dumps(obj))
    _write(path, lines)


def code_task(path, count: int = 20000, seed: int = 1234) -> None:
    rng = random.Random(seed)
    lines = []
    for _ in range(count):
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        op = rng.choice(["+", "-", "*"])
        result = a + b if op == "+" else a - b if op == "-" else a * b
        lines.append(f"def f(a, b):\n    return a {op} b\n# f({a}, {b}) = {result}")
    _write(path, lines)


def translate_task(path, count: int = 20000, seed: int = 1234) -> None:
    rng = random.Random(seed)
    words = {"hello": "hej", "world": "världen", "cat": "katt", "dog": "hund",
             "house": "hus", "car": "bil", "book": "bok", "water": "vatten",
             "good": "bra", "night": "natt", "friend": "vän", "day": "dag"}
    lines = []
    for _ in range(count):
        english, swedish = rng.choice(list(words.items()))
        lines.append(f"English: {english} Swedish: {swedish}")
    _write(path, lines)


GENERATORS = {
    "arithmetic": arithmetic,
    "copy": copy_task,
    "reverse": reverse_task,
    "json": json_task,
    "code": code_task,
    "translate": translate_task,
}


def generate(name: str, path, count: int = 20000, seed: int = 1234) -> None:
    """Dispatch a named task generator to a text file."""
    if name not in GENERATORS:
        raise ValueError(f"unknown task {name!r}; choose from {sorted(GENERATORS)}")
    GENERATORS[name](path, count=count, seed=seed)
