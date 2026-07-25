"""Run the Bullshit-Bench prompt set against a list of models.

Loads the trick prompts, queries each configured model once per prompt, scores
the responses, and writes everything to ``results/run_<timestamp>.json``.

Usage::

    python run_bench.py                          # default model roster
    python run_bench.py --models gpt-4o gemini-2.0-flash
    python run_bench.py --list-models            # show the registry
    python run_bench.py --dry-run                # no API calls at all, canned answers
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

from dotenv import load_dotenv

import providers
import scorer

load_dotenv()

ROOT = Path(__file__).resolve().parent
PROMPTS_PATH = ROOT / "prompts" / "trick_prompts.json"
RESULTS_DIR = ROOT / "results"

T = TypeVar("T")

# Retry tuning. Providers surface rate limits as their own exception types, so
# we retry on anything that is not explicitly marked non-retryable.
MAX_ATTEMPTS = 5
BASE_DELAY = 2.0
MAX_DELAY = 60.0

# Deterministic failures: a second attempt cannot change the outcome, so raise
# immediately instead of burning four more calls and ~60s of backoff.
_NON_RETRYABLE = (
    providers.MissingCredentials,
    providers.MissingDependency,
    providers.UnknownModel,
    providers.EmptyResponse,
)


def load_prompts(path: Path = PROMPTS_PATH) -> List[Dict[str, str]]:
    """Read the prompt set and validate its shape.

    Args:
        path: Path to a JSON list of ``{id, prompt, type}`` objects.

    Returns:
        The prompt list, in file order.

    Raises:
        SystemExit: The file is missing, malformed, or has duplicate ids.
    """
    if not path.exists():
        sys.exit(f"Prompt file not found: {path}")

    try:
        prompts = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.exit(f"{path.name} is not valid JSON: {exc}")

    if not isinstance(prompts, list) or not prompts:
        sys.exit(f"{path.name} must be a non-empty JSON list")

    seen: set[str] = set()
    for i, item in enumerate(prompts):
        missing = {"id", "prompt", "type"} - set(item)
        if missing:
            sys.exit(f"{path.name}[{i}] is missing field(s): {', '.join(sorted(missing))}")
        if item["id"] in seen:
            sys.exit(f"{path.name}: duplicate prompt id {item['id']!r}")
        seen.add(item["id"])

    return prompts


def with_retry(
    fn: Callable[[], T],
    *,
    description: str,
    max_attempts: int = MAX_ATTEMPTS,
    verbose: bool = True,
) -> T:
    """Call ``fn``, retrying transient failures with exponential backoff.

    Credential and dependency errors are raised immediately — retrying a missing
    API key just wastes wall-clock time.

    Args:
        fn: Zero-argument callable to invoke.
        description: Human-readable label used in retry messages.
        max_attempts: Total attempts, including the first.
        verbose: Print a line per retry.

    Returns:
        Whatever ``fn`` returns.

    Raises:
        Exception: The last failure, once attempts are exhausted.
    """
    last_error: Optional[BaseException] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except _NON_RETRYABLE:
            raise
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise their own types
            last_error = exc
            if attempt == max_attempts:
                break
            delay = min(MAX_DELAY, BASE_DELAY * 2 ** (attempt - 1))
            delay += random.uniform(0, delay * 0.25)  # jitter to desynchronise retries
            if verbose:
                print(
                    f"    ! {description} failed ({type(exc).__name__}: {exc}); "
                    f"retry {attempt}/{max_attempts - 1} in {delay:.1f}s",
                    file=sys.stderr,
                )
            time.sleep(delay)

    assert last_error is not None
    raise last_error


def run_benchmark(
    model_keys: Sequence[str],
    prompts: Sequence[Dict[str, str]],
    *,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    judge_model: Optional[str] = None,
    do_score: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Query every model with every prompt and score the answers.

    Args:
        model_keys: Registry keys of the models to benchmark.
        prompts: Prompt records from :func:`load_prompts`.
        temperature: Sampling temperature, dropped for models that reject it.
        max_tokens: Output cap per response.
        judge_model: Judge registry key, or ``None`` to auto-resolve.
        do_score: Set ``False`` to collect raw responses without grading.
        dry_run: Skip every API call - both generation and judging - and emit
            placeholder text scored by the heuristic. Exercises the whole
            pipeline for free.

    Returns:
        A run record: ``{run_id, started_at, config, records}``.
    """
    # A dry run must not spend money anywhere, and the judge is an API call too:
    # grade with the heuristic scorer instead of quietly billing for 1 judge call
    # per response.
    resolved_judge = (
        scorer.resolve_judge(judge_model) if do_score and not dry_run else None
    )
    if do_score:
        if dry_run:
            print("Judge: skipped for --dry-run - scoring with the heuristic.")
        elif resolved_judge:
            print(f"Judge: {resolved_judge}")
        else:
            print("Judge: none configured - falling back to the heuristic scorer.")

    started = datetime.now(timezone.utc)
    records: List[Dict[str, Any]] = []

    for model_key in model_keys:
        spec = providers.get_model(model_key)
        print(f"\n=== {spec.label} ({model_key}) ===")

        for item in prompts:
            label = f"{model_key} / {item['id']}"
            print(f"  - {item['id']}", end="", flush=True)

            record: Dict[str, Any] = {
                "model": model_key,
                "model_label": spec.label,
                "provider": spec.provider,
                "prompt_id": item["id"],
                "prompt_type": item["type"],
                "prompt": item["prompt"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "response": None,
                "error": None,
                "latency_s": None,
                "scores": None,
            }

            if dry_run:
                record["response"] = f"[dry-run placeholder response for {item['id']}]"
                record["latency_s"] = 0.0
            else:
                start = time.perf_counter()
                try:
                    record["response"] = with_retry(
                        lambda: providers.generate(
                            model_key,
                            item["prompt"],
                            temperature=temperature,
                            max_tokens=max_tokens,
                        ),
                        description=label,
                    )
                except Exception as exc:  # noqa: BLE001 - record and keep going
                    record["error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    record["latency_s"] = round(time.perf_counter() - start, 3)

            if record["error"]:
                print(f"  FAILED: {record['error']}")
                records.append(record)
                continue

            if do_score:
                try:
                    # `scorer.score(judge_model=None)` means "auto-resolve", which
                    # would reach for a judge during a dry run - so call the
                    # heuristic directly instead of relying on the None default.
                    grade = (
                        (lambda: scorer.score_heuristic(
                            item["prompt"], record["response"], item["type"]
                        ))
                        if dry_run
                        else (lambda: scorer.score(
                            item["prompt"],
                            record["response"],
                            item["type"],
                            judge_model=resolved_judge,
                        ))
                    )
                    result = with_retry(grade, description=f"judge {label}")
                    record["scores"] = result.to_dict()
                    print(f"  scored {result.total:.1f}/10")
                except Exception as exc:  # noqa: BLE001
                    record["scores"] = None
                    record["error"] = f"scoring failed: {type(exc).__name__}: {exc}"
                    print(f"  SCORING FAILED: {exc}")
            else:
                print("  ok")

            records.append(record)

    return {
        "run_id": started.strftime("%Y%m%dT%H%M%SZ"),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "models": list(model_keys),
            "prompt_count": len(prompts),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "judge_model": resolved_judge,
            "scored": do_score,
            "dry_run": dry_run,
        },
        "records": records,
    }


def save_run(run: Dict[str, Any], results_dir: Path = RESULTS_DIR) -> Path:
    """Write a run record to ``results/run_<timestamp>.json``."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"run_{run['run_id']}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _print_registry() -> None:
    print(f"{'KEY':<22} {'PROVIDER':<11} {'KEY SET':<8} DEFAULT  LABEL")
    available = set(providers.available_model_keys())
    for key, spec in providers.MODELS.items():
        print(
            f"{key:<22} {spec.provider:<11} "
            f"{'yes' if key in available else 'no':<8} "
            f"{'yes' if spec.enabled_by_default else 'no':<8} {spec.label}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Bullshit-Bench prompt set.")
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="KEY",
        help="Model keys to benchmark (default: the registry's default roster).",
    )
    parser.add_argument(
        "--prompts", type=Path, default=PROMPTS_PATH, help="Path to the prompt JSON file."
    )
    parser.add_argument(
        "--out", type=Path, default=RESULTS_DIR, help="Directory for run output."
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Output cap per response. Models that think by default spend this "
             "budget on reasoning before any visible text, so keep it generous.",
    )
    parser.add_argument(
        "--judge", metavar="KEY", help="Judge model key (default: auto-resolve)."
    )
    parser.add_argument(
        "--no-score", action="store_true", help="Collect raw responses without grading them."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip every API call (generation and judging); emit placeholder "
             "responses scored by the heuristic.",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="Print the model registry and exit."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_models:
        _print_registry()
        return 0

    prompts = load_prompts(args.prompts)

    model_keys = args.models or providers.default_model_keys()
    if not model_keys:
        sys.exit("No models selected. Pass --models, or mark models enabled_by_default.")

    for key in model_keys:
        providers.get_model(key)  # fail fast on a typo'd key

    if not args.dry_run:
        have_keys = set(providers.available_model_keys())
        unusable = [k for k in model_keys if k not in have_keys]
        if unusable:
            print(
                f"Warning: no API key for {', '.join(unusable)}. "
                "Those models will be recorded as errors.",
                file=sys.stderr,
            )

    print(f"Running {len(prompts)} prompts x {len(model_keys)} models "
          f"= {len(prompts) * len(model_keys)} calls")

    run = run_benchmark(
        model_keys,
        prompts,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        judge_model=args.judge,
        do_score=not args.no_score,
        dry_run=args.dry_run,
    )
    path = save_run(run, args.out)

    ok = sum(1 for r in run["records"] if r["response"] and not r["error"])
    print(f"\nSaved {len(run['records'])} records ({ok} successful) to {path}")
    print("Next: python leaderboard.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
