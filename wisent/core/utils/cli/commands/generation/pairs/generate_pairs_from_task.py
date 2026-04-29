"""Generate pairs from task command execution logic."""

import sys
import json
import os

from wisent.core.utils.config_tools.constants import JSON_INDENT


def _is_rate_limit_exc(exc):
    """True if exc OR any link in its __cause__/__context__ chain is a 429.

    wisent.extractors.lm_eval.lm_task_pairs_generation has bare
    'except Exception:' blocks that swallow HF 429s into a downstream
    NoDocsAvailableError. The 429 is preserved on __context__, so walk
    the chain so the retry below sees through the wrap.
    """
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        msg = str(cur).lower()
        if "429" in msg or "too many requests" in msg or "rate limit" in msg:
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return False


def execute_generate_pairs_from_task(args):
    """Execute the generate-pairs-from-task command - load and save contrastive pairs from a task."""
    # Expand task if it's a skill or risk name
    from wisent.core.control.tasks.base.task_selector import expand_task_if_skill_or_risk
    if hasattr(args, 'task_name') and args.task_name:
        args.task_name = expand_task_if_skill_or_risk(args.task_name)
    from wisent.core.utils.services.benchmarks import validate_benchmark
    validate_benchmark(args.task_name, allow_subtasks=getattr(args, 'allow_subtasks', False))

    from wisent.extractors.lm_eval.lm_task_pairs_generation import (
        build_contrastive_pairs,
    )

    print(f"\n📊 Generating contrastive pairs from task: {args.task_name}")

    pairs = None
    pairs_task_name = args.task_name
    # Fast path: pull pre-computed pair_texts from wisent-ai/activations on HF
    # before falling back to the lm-eval path that hits HF dataset_info per
    # task and gets 429-throttled when multiple agents run in parallel. The
    # uploaded file already matches this command's OUTPUT schema
    # ({task_name, num_pairs, pairs:[pair.to_dict(),...]}), so we copy it
    # straight to args.output instead of round-tripping through
    # ContrastivePair construction. Cache miss / network error falls
    # through to the fresh-build path. 429 retries with jittered
    # exponential backoff because falling through hits the SAME rate-
    # limited token bucket via lm-eval.
    try:
        from huggingface_hub import hf_hub_download
        import shutil as _shutil
        import time as _time
        import random as _random

        cached_path = None
        for _attempt in range(8):
            try:
                cached_path = hf_hub_download(
                    repo_id="wisent-ai/activations",
                    repo_type="dataset",
                    filename=f"pair_texts/{args.task_name}.json",
                    token=os.environ.get("HF_TOKEN") or None,
                )
                break
            except Exception as _exc:
                _msg = str(_exc).lower()
                _is_429 = _is_rate_limit_exc(_exc)
                _is_404 = "404" in _msg or "not found" in _msg or "entrynotfounderror" in _msg
                if _is_404:
                    print(f"   ⚠ pair_texts cache MISS (404) for '{args.task_name}'; "
                          f"will fresh-build", flush=True)
                    cached_path = None
                    break
                if _is_429 and _attempt < 7:
                    _base = 30 * (2 ** min(_attempt, 5))
                    _time.sleep(min(_base + _random.uniform(0, _base), 600))
                    continue
                print(f"   ⚠ pair_texts cache fetch failed for '{args.task_name}': {_exc}",
                      flush=True)
                cached_path = None
                break

        if cached_path:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            _shutil.copyfile(cached_path, args.output)
            with open(cached_path, "r") as _f:
                _doc = json.load(_f)
            _n = _doc.get("num_pairs", len(_doc.get("pairs", [])))
            print(f"   ✓ Copied {_n} pre-computed pairs from HF cache to {args.output}")
            print(f"\n✅ Contrastive pairs generation completed successfully!\n")
            return
    except Exception as exc:
        print(f"   ⚠ HF pair_texts cache fast-path crashed for '{args.task_name}': {exc}; "
              f"falling back to fresh build", flush=True)

    try:
        if pairs is None:
            print(f"\n🔄 Loading task '{args.task_name}'...")
            print(f"   🔨 Building contrastive pairs...")
            # Retry on 429 here too: lm-eval's task instantiation
            # (loader.load_lm_eval_task -> get_task_dict) hits HF
            # dataset_info BEFORE the per-split docs fetch the
            # dataset_splits.get_all_docs_from_task retry handles, so a
            # 429 at task-load time never reaches that retry layer.
            import time as _t, random as _r
            for _attempt in range(8):
                try:
                    pairs = build_contrastive_pairs(
                        task_name=args.task_name,
                        limit=getattr(args, 'limit', None),
                        train_ratio=args.train_ratio,
                    )
                    break
                except Exception as _exc:
                    _is_429 = _is_rate_limit_exc(_exc)
                    if _is_429 and _attempt < 7:
                        _base = 30 * (2 ** min(_attempt, 5))
                        _t.sleep(min(_base + _r.uniform(0, _base), 600))
                        continue
                    raise

        print(f"   ✓ Generated {len(pairs)} contrastive pairs")

        # 3. Convert pairs to dict format for JSON serialization
        print(f"\n💾 Saving pairs to '{args.output}'...")
        pairs_data = []
        for pair in pairs:
            pair_dict = pair.to_dict()
            pairs_data.append(pair_dict)

        # 4. Save to JSON file
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({
                'task_name': pairs_task_name,
                'num_pairs': len(pairs),
                'pairs': pairs_data
            }, f, indent=JSON_INDENT)

        print(f"   ✓ Saved {len(pairs)} pairs to: {args.output}")
        print(f"\n✅ Contrastive pairs generation completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        raise
