"""Generate pairs from task command execution logic."""

import sys
import json
import os

from wisent.core.utils.config_tools.constants import JSON_INDENT


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
    # (uploaded once by upload_pair_texts) before falling back to the lm-eval
    # path that hits HF dataset_info per task and gets 429-throttled when
    # multiple agents run in parallel. Cache miss / network error falls
    # through to the fresh-build path.
    try:
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
            load_pair_texts_from_hf,
        )
        from wisent.core.primitives.contrastive_pairs.core.buliders import (
            from_phrase_pairs,
        )
        cached = load_pair_texts_from_hf(
            args.task_name,
            limit=getattr(args, "limit", 0) or 0,
            use_cache=True,
        )
        if cached:
            print(f"   ✓ Loaded {len(cached)} pre-computed pairs from HF cache")
            phrase_pairs = [
                {"prompt": v.get("prompt", ""),
                 "positive": v.get("positive", ""),
                 "negative": v.get("negative", "")}
                for v in cached.values()
            ]
            cps = from_phrase_pairs(phrase_pairs)
            pairs = list(cps.pairs)
    except Exception as exc:
        print(f"   ⚠ HF pair_texts cache miss for '{args.task_name}': {exc}; "
              f"falling back to fresh build", flush=True)

    try:
        if pairs is None:
            print(f"\n🔄 Loading task '{args.task_name}'...")
            print(f"   🔨 Building contrastive pairs...")
            pairs = build_contrastive_pairs(
                task_name=args.task_name,
                limit=getattr(args, 'limit', None),
                train_ratio=args.train_ratio,
            )

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
