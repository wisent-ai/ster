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
    # task and gets 429-throttled when multiple agents run in parallel.
    # WISENT_DISABLE_PAIR_CACHE: skip this fast path so re-runs always
    # produce a fresh organic extraction (no cache pollution).
    _skip_fast_path = bool(os.environ.get("WISENT_DISABLE_PAIR_CACHE"))
    try:
        if _skip_fast_path:
            raise RuntimeError("WISENT_DISABLE_PAIR_CACHE set; skipping HF fast path")
        from huggingface_hub import HfApi, hf_hub_download
        import shutil as _shutil
        import time as _time
        import random as _random

        cached_path = None
        try:
            _rel = f"pair_texts/{args.task_name}.json"
            _free = _shutil.disk_usage(os.path.expanduser("~")).free
            _headroom = 5 * 1024 ** 3
            _entries = HfApi().list_repo_tree(
                "wisent-ai/activations", repo_type="dataset",
                path_in_repo=_rel, token=os.environ.get("HF_TOKEN") or None)
            _size = next((getattr(e, "size", None) for e in _entries
                          if getattr(e, "path", "") == _rel), None)
            if _size and _size > max(0, _free - _headroom):
                print(f"   ⚠ pair_texts cache {_rel} is {_size/(1024**3):.1f}GB "
                      f"but home cache has only {_free/(1024**3):.1f}GB free; "
                      "fresh-building instead", flush=True)
                raise RuntimeError("pair_texts cache too large for local disk")
        except RuntimeError:
            raise
        except Exception:
            pass
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
            with open(cached_path, "r") as _f:
                _doc = json.load(_f)
            _n = _doc.get("num_pairs", len(_doc.get("pairs", [])))
            # Stale empty cache entries (uploaded by an earlier crashed run)
            # otherwise propagate as "Loaded 0 pairs" downstream and the
            # whole job exits non-zero with no useful output.
            if _n <= 0:
                print(f"   ⚠ HF cache for '{args.task_name}' has 0 pairs; "
                      f"forcing fresh build", flush=True)
            elif (
                getattr(args, 'limit', None) is not None
                and args.limit > 0
                and _n < args.limit
            ):
                # Cache file has fewer pairs than the requested --limit.
                # For LEAF tasks this is fine (the source dataset itself
                # is small) — but for GROUP tasks (bigbench, mmlu_prox,
                # leaderboard, etc.) it means the cache is a sparse
                # sample, NOT the full subtask union, and the activation
                # extractor would run on too few pairs. Detect "group"
                # by counting unique subtask identifiers across the
                # cache; any value > 1 means multi-subtask data and
                # forces a fresh lm-eval-harness build that enumerates
                # all subtasks and returns up to `limit` pairs
                # distributed across them.
                _pairs = _doc.get("pairs", [])
                _unique_subtasks: set = set()
                for _p in _pairs:
                    _md = _p.get("metadata") if isinstance(_p, dict) else None
                    if not isinstance(_md, dict):
                        _md = {}
                    _k = (
                        _p.get("trait_label")
                        or _p.get("task_name")
                        or _p.get("subtask")
                        or _md.get("source_task")
                        or _md.get("task")
                        or _md.get("subtask")
                    )
                    if _k:
                        _unique_subtasks.add(_k)
                if len(_unique_subtasks) > 1:
                    print(
                        f"   ⚠ HF cache for '{args.task_name}' has only {_n} pairs "
                        f"across {len(_unique_subtasks)} subtasks (limit={args.limit}); "
                        f"forcing fresh lm-eval build to enumerate full subtask tree",
                        flush=True,
                    )
                    # Don't return; fall through to the lm-eval build
                    # path at line 158 (same control-flow as _n <= 0).
                else:
                    # Genuine small-leaf case: the source dataset just
                    # doesn't have more pairs. Keep the cache as-is.
                    os.makedirs(
                        os.path.dirname(os.path.abspath(args.output)),
                        exist_ok=True,
                    )
                    _shutil.copyfile(cached_path, args.output)
                    print(
                        f"   ✓ Copied {_n} pre-computed pairs from HF cache to "
                        f"{args.output} (single-subtask leaf, {_n} < limit "
                        f"{args.limit} is intrinsic)"
                    )
                    print(f"\n✅ Contrastive pairs generation completed successfully!\n")
                    return
            else:
                # Apply --limit to cached pairs. Earlier this used
                # _shutil.copyfile which blindly copied the entire cache file
                # regardless of args.limit, producing 1997-pair jobs that
                # took hours when the caller asked for 1 pair (and any
                # smaller limit). Confirmed live on 2026-05-08: NTREX jobs
                # submitted with --limit 1 were processing 1997 pairs each.
                # When the cache contains pairs from multiple sub/sub-subtasks
                # (each pair carries trait_label or task_name in the pair
                # dict), distribute the limit evenly across them so the
                # sample is balanced rather than just the first N.
                _limit = getattr(args, 'limit', None)
                _pairs = _doc.get("pairs", [])
                if _limit is not None and _limit > 0 and len(_pairs) > _limit:
                    _by_grp: dict[str, list] = {}
                    for _p in _pairs:
                        _md = _p.get("metadata") if isinstance(_p, dict) else None
                        if not isinstance(_md, dict):
                            _md = {}
                        # Older HF caches (e.g. pair_texts/bigbench.json
                        # written 2026-04) carry the subtask identifier
                        # only inside metadata.source_task; later writers
                        # promote it to a top-level key. Look at all of
                        # them so the distribution logic actually buckets
                        # by subtask for cache files written either way.
                        _grp = (_p.get("trait_label")
                                or _p.get("task_name")
                                or _p.get("subtask")
                                or _md.get("source_task")
                                or _md.get("task")
                                or _md.get("subtask")
                                or "")
                        _by_grp.setdefault(_grp, []).append(_p)
                    _picked: list = []
                    if len(_by_grp) > 1:
                        _per_grp = max(1, _limit // len(_by_grp))
                        for _grp, _grp_pairs in _by_grp.items():
                            _picked.extend(_grp_pairs[:_per_grp])
                            if len(_picked) >= _limit:
                                break
                        _picked = _picked[:_limit]
                    else:
                        _picked = _pairs[:_limit]
                    _doc["pairs"] = _picked
                    _doc["num_pairs"] = len(_picked)
                    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                    with open(args.output, "w") as _of:
                        json.dump(_doc, _of, indent=JSON_INDENT)
                    print(f"   ✓ Copied {len(_picked)}/{_n} pre-computed pairs from HF cache "
                          f"(applied --limit {_limit}, {len(_by_grp)} groups) to {args.output}")
                else:
                    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                    _shutil.copyfile(cached_path, args.output)
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
