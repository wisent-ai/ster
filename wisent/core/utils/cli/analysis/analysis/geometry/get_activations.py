"""Get activations command execution logic."""

import sys
import json
import os
import time

from wisent.core.control.steering_methods.configs.optimal import get_optimal, get_optimal_extraction_strategy
from wisent.core.utils.config_tools.constants import (
    JSON_INDENT, PROGRESS_LOG_INTERVAL_10, ARCHITECTURE_MODULE_LIMIT_DEFAULT,
)


def execute_get_activations(args, *, architecture_module_limit: int = ARCHITECTURE_MODULE_LIMIT_DEFAULT):
    """Execute the get-activations command - load pairs and collect activations."""
    from wisent.core.primitives.models.wisent_model import WisentModel
    from wisent.core.primitives.model_interface.core.activations.activations_collector import ActivationCollector
    from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy, ExtractionComponent
    
    from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
    from wisent.core.primitives.contrastive_pairs.core.io.response import PositiveResponse, NegativeResponse
    from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet

    raw_mode = getattr(args, 'raw', False)
    mode_str = "raw hidden states" if raw_mode else "activations"
    
    print(f"\n🎨 Collecting {mode_str} from contrastive pairs")
    print(f"   Input file: {args.pairs_file}")
    print(f"   Model: {args.model}")
    if raw_mode:
        print(f"   Mode: RAW (full sequences, reusable for multiple strategies)")

    start_time = time.time() if args.timing else None

    try:
        # 1. Load pairs from JSON
        print(f"\n📂 Loading contrastive pairs...")
        if not os.path.exists(args.pairs_file):
            raise FileNotFoundError(f"Pairs file not found: {args.pairs_file}")

        with open(args.pairs_file, 'r') as f:
            data = json.load(f)

        # Handle both formats: dict with 'pairs' key or direct list
        if isinstance(data, dict):
            pairs_list = data.get('pairs', [])
            task_name = data.get('task_name', 'unknown')
            trait_label = data.get('trait_label', task_name)
        else:
            pairs_list = data
            task_name = 'unknown'
            trait_label = 'unknown'

        print(f"   ✓ Loaded {len(pairs_list)} pairs")

        # 2. Load model (or use cached)
        cached_model = getattr(args, 'cached_model', None)
        if cached_model is not None:
            model = cached_model
            print(f"\n🤖 Using cached model ({model.num_layers} layers)")
        else:
            print(f"\n🤖 Loading model '{args.model}'...")
            model = WisentModel(args.model, device=args.device)
            print(f"   ✓ Model loaded with {model.num_layers} layers")
        num_attention_heads = getattr(model.hf_model.config, 'num_attention_heads', None)
        num_kv_heads = getattr(model.hf_model.config, 'num_key_value_heads', num_attention_heads)

        # 3. Determine layers to collect (1-indexed for API)
        if args.layers is None:
            # Default: use ALL layers (1-indexed: 1..num_layers)
            layers = list(range(1, model.num_layers + 1))
        elif args.layers.lower() == 'all':
            # Use all layers (1-indexed: 1..num_layers)
            layers = list(range(1, model.num_layers + 1))
        else:
            layers = [int(l.strip()) for l in args.layers.split(',')]

        # Convert to strings for API (1-indexed)
        layer_strs = [str(l) for l in layers]

        print(f"\n🎯 Collecting activations from {len(layers)} layer(s): {layers}")

        # 4. Get extraction strategy from args
        extraction_strategy = ExtractionStrategy(getattr(args, 'extraction_strategy', get_optimal_extraction_strategy()))
        extraction_component = ExtractionComponent(getattr(args, 'extraction_component', get_optimal("extraction_component")))
        print(f"   Extraction strategy: {extraction_strategy.value}")
        print(f"   Extraction component: {extraction_component.value}")

        # 5. Create pair set and reconstruct pairs
        pair_set = ContrastivePairSet(name=task_name, task_type=trait_label)

        for pair_data in pairs_list:
            pair = ContrastivePair(
                prompt=pair_data['prompt'],
                positive_response=PositiveResponse(
                    model_response=pair_data['positive_response']['model_response']
                ),
                negative_response=NegativeResponse(
                    model_response=pair_data['negative_response']['model_response']
                ),
                label=pair_data.get('label', trait_label),
                trait_description=pair_data.get('trait_description', ''),
            )
            pair_set.add(pair)

        # 6. Collect activations
        collector = ActivationCollector(model=model, architecture_module_limit=architecture_module_limit)

        if raw_mode:
            # RAW MODE: Collect full hidden states [seq_len, hidden_size]
            print(f"\n⚡ Collecting RAW hidden states...")
            from wisent.core.primitives.model_interface.core.activations.activation_cache import get_strategy_text_family
            text_family = get_strategy_text_family(extraction_strategy)
            
            raw_pairs_data = []
            for i, pair in enumerate(pair_set.pairs):
                if args.verbose or (i + 1) % PROGRESS_LOG_INTERVAL_10 == 0:
                    print(f"   Processing pair {i+1}/{len(pair_set.pairs)}...", end='\r', flush=True)

                # Collect RAW hidden states (full sequences)
                raw_data = collector.collect_raw(
                    pair, strategy=extraction_strategy,
                    layers=layer_strs, component=extraction_component,
                )
                raw_pairs_data.append({
                    'pair': pair,
                    'raw_data': raw_data,
                })

            print(f"   ✓ Collected raw hidden states for {len(raw_pairs_data)} pairs")

            # Convert to JSON format (raw mode)
            print(f"\n💾 Saving raw activations to '{args.output}'...")
            output_data = {
                'task_name': task_name,
                'trait_label': trait_label,
                'model': args.model,
                'layers': layers,
                'extraction_strategy': extraction_strategy.value,
                'extraction_component': extraction_component.value,
                'text_family': text_family,
                'raw_mode': True,
                'num_pairs': len(raw_pairs_data),
                'pairs': []
            }

            for item in raw_pairs_data:
                pair = item['pair']
                raw_data = item['raw_data']
                
                pair_dict = {
                    'prompt': pair.prompt,
                    'positive_response': {
                        'model_response': pair.positive_response.model_response,
                        'answer_text': raw_data['pos_answer_text'],
                        'prompt_len': raw_data['pos_prompt_len'],
                        'layers_hidden_states': {}
                    },
                    'negative_response': {
                        'model_response': pair.negative_response.model_response,
                        'answer_text': raw_data['neg_answer_text'],
                        'prompt_len': raw_data['neg_prompt_len'],
                        'layers_hidden_states': {}
                    },
                    'label': pair.label,
                    'trait_description': pair.trait_description,
                }

                # Convert raw hidden states to lists
                for layer_str, hs in raw_data['pos_hidden_states'].items():
                    pair_dict['positive_response']['layers_hidden_states'][layer_str] = hs.cpu().tolist()
                for layer_str, hs in raw_data['neg_hidden_states'].items():
                    pair_dict['negative_response']['layers_hidden_states'][layer_str] = hs.cpu().tolist()

                output_data['pairs'].append(pair_dict)

        else:
            # STANDARD MODE: Collect extracted vectors [hidden_size]
            print(f"\n⚡ Collecting activations...")
            
            enriched_pairs = []
            qk_data_per_pair = []
            layer_norms = {l: [] for l in layer_strs}
            for i, pair in enumerate(pair_set.pairs):
                if args.verbose:
                    print(f"   Processing pair {i+1}/{len(pair_set.pairs)}...")
                updated_pair = collector.collect(
                    pair, strategy=extraction_strategy,
                    layers=layer_strs, component=extraction_component,
                    capture_qk=True,
                )
                enriched_pairs.append(updated_pair)
                qk_data_per_pair.append(dict(collector._last_qk))
                
                # Record norms for calibration
                for layer_str in layer_strs:
                    if updated_pair.positive_response.layers_activations:
                        pos_act = updated_pair.positive_response.layers_activations.get(layer_str)
                        if pos_act is not None:
                            layer_norms[layer_str].append(pos_act.norm().item())
                    if updated_pair.negative_response.layers_activations:
                        neg_act = updated_pair.negative_response.layers_activations.get(layer_str)
                        if neg_act is not None:
                            layer_norms[layer_str].append(neg_act.norm().item())

            print(f"   ✓ Collected activations for {len(enriched_pairs)} pairs")
            
            # Compute calibration norms (average per layer)
            calibration_norms = {}
            for layer_str, norms in layer_norms.items():
                if norms:
                    calibration_norms[layer_str] = sum(norms) / len(norms)
            
            if calibration_norms:
                print(f"   📊 Calibration norms: {', '.join(f'L{k}={v:.1f}' for k, v in calibration_norms.items())}")

            # Convert to JSON format (standard mode)
            print(f"\n💾 Saving enriched pairs to '{args.output}'...")
            output_data = {
                'task_name': task_name,
                'trait_label': trait_label,
                'model': args.model,
                'layers': layers,
                'extraction_strategy': extraction_strategy.value,
                'extraction_component': extraction_component.value,
                'raw_mode': False,
                'num_pairs': len(enriched_pairs),
                'calibration_norms': calibration_norms,
                'num_attention_heads': num_attention_heads,
                'num_key_value_heads': num_kv_heads,
                'pairs': []
            }

            for i, pair in enumerate(enriched_pairs):
                pair_dict = {
                    'prompt': pair.prompt,
                    'positive_response': {
                        'model_response': pair.positive_response.model_response,
                        'layers_activations': {}
                    },
                    'negative_response': {
                        'model_response': pair.negative_response.model_response,
                        'layers_activations': {}
                    },
                    'label': pair.label,
                    'trait_description': pair.trait_description,
                }
                if pair.positive_response.layers_activations:
                    for layer_str, act in pair.positive_response.layers_activations.items():
                        if act is not None:
                            pair_dict['positive_response']['layers_activations'][layer_str] = act.cpu().tolist()
                if pair.negative_response.layers_activations:
                    for layer_str, act in pair.negative_response.layers_activations.items():
                        if act is not None:
                            pair_dict['negative_response']['layers_activations'][layer_str] = act.cpu().tolist()
                # Serialize Q/K projections
                qk = qk_data_per_pair[i]
                for side, side_key in [("positive_response", "pos"), ("negative_response", "neg")]:
                    side_qk = qk.get(side_key, {})
                    q_acts, k_acts = {}, {}
                    for key, val in side_qk.items():
                        layer = key[2:]
                        (q_acts if key.startswith("q_") else k_acts)[layer] = val.cpu().tolist()
                    pair_dict[side]['q_proj_activations'] = q_acts
                    pair_dict[side]['k_proj_activations'] = k_acts
                output_data['pairs'].append(pair_dict)

        # 8. Save to file
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=JSON_INDENT)

        print(f"   ✓ Saved enriched pairs to: {args.output}")

        if args.timing:
            elapsed = time.time() - start_time
            print(f"   ⏱️  Total time: {elapsed:.2f}s")

        print(f"\n✅ Activation collection completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        raise


def execute_get_activations_multi(
    *,
    pairs_file: str,
    output_dir: str,
    model: object,
    model_name: str,
    device: str,
    layers: str,
    strategies: list[str],
    component: str = "residual_stream",
    architecture_module_limit: int = ARCHITECTURE_MODULE_LIMIT_DEFAULT,
    capture_qk: bool = True,
    verbose: bool = False,
) -> dict[str, str]:
    """Run all strategies in `strategies` against a shared forward pass.

    The strategies MUST belong to the same text family (caller's
    responsibility — see CachedActivations.get_strategy_text_family).

    For each strategy, writes the standard get-activations JSON output
    (byte-compatible with execute_get_activations(non-raw)) to
    `<output_dir>/<task_name>__<strategy>.json`.

    Returns: {strategy: output_path}.

    Used by extract_and_upload to fold the 5 chat_* strategies into ONE
    forward pass per pair (and one forward pass each for mc_balanced and
    role_play). End-to-end this cuts forward-pass count from 7 to 3 per
    task.
    """
    from wisent.core.primitives.model_interface.core.activations.activations_collector import ActivationCollector
    from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy, ExtractionComponent
    from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
    from wisent.core.primitives.contrastive_pairs.core.io.response import PositiveResponse, NegativeResponse
    from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet

    if not strategies:
        return {}

    if verbose:
        print(f"[multi] strategies={strategies} (one forward pass per pair)", flush=True)

    with open(pairs_file, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        pairs_list = data.get("pairs", [])
        task_name = data.get("task_name", "unknown")
        trait_label = data.get("trait_label", task_name)
    else:
        pairs_list = data
        task_name = "unknown"
        trait_label = "unknown"

    strategy_objs = [ExtractionStrategy(s) for s in strategies]
    component_obj = ExtractionComponent(component)

    if layers is None or str(layers).lower() == "all":
        layer_ints = list(range(1, model.num_layers + 1))
    else:
        layer_ints = [int(x.strip()) for x in str(layers).split(",")]
    layer_strs = [str(l) for l in layer_ints]

    num_attention_heads = getattr(model.hf_model.config, "num_attention_heads", None)
    num_kv_heads = getattr(model.hf_model.config, "num_key_value_heads", num_attention_heads)

    pair_set = ContrastivePairSet(name=task_name, task_type=trait_label)
    for pair_data in pairs_list:
        pair = ContrastivePair(
            prompt=pair_data["prompt"],
            positive_response=PositiveResponse(
                model_response=pair_data["positive_response"]["model_response"],
            ),
            negative_response=NegativeResponse(
                model_response=pair_data["negative_response"]["model_response"],
            ),
            label=pair_data.get("label", trait_label),
            trait_description=pair_data.get("trait_description", ""),
        )
        pair_set.add(pair)

    collector = ActivationCollector(
        model=model, architecture_module_limit=architecture_module_limit,
    )

    enriched_per_strategy: dict[str, list] = {s: [] for s in strategies}
    qk_per_strategy: dict[str, list] = {s: [] for s in strategies}

    for i, pair in enumerate(pair_set.pairs):
        if verbose or (i + 1) % PROGRESS_LOG_INTERVAL_10 == 0:
            print(f"   pair {i+1}/{len(pair_set.pairs)}...", end="\r", flush=True)
        per_strat = collector.collect_multi_strategy(
            pair, strategy_objs, layers=layer_strs,
            component=component_obj, capture_qk=capture_qk,
        )
        for s_obj in strategy_objs:
            enriched_per_strategy[s_obj.value].append(per_strat[s_obj]["pair"])
            qk_per_strategy[s_obj.value].append(per_strat[s_obj]["qk"])

    os.makedirs(output_dir, exist_ok=True)
    written: dict[str, str] = {}
    for strategy in strategies:
        enriched_pairs = enriched_per_strategy[strategy]
        qk_data_per_pair = qk_per_strategy[strategy]

        layer_norms = {l: [] for l in layer_strs}
        for ep in enriched_pairs:
            for layer_str in layer_strs:
                if ep.positive_response.layers_activations:
                    pa = ep.positive_response.layers_activations.get(layer_str)
                    if pa is not None:
                        layer_norms[layer_str].append(pa.norm().item())
                if ep.negative_response.layers_activations:
                    na = ep.negative_response.layers_activations.get(layer_str)
                    if na is not None:
                        layer_norms[layer_str].append(na.norm().item())
        calibration_norms = {
            l: (sum(v) / len(v)) for l, v in layer_norms.items() if v
        }

        output_data = {
            "task_name": task_name,
            "trait_label": trait_label,
            "model": model_name,
            "layers": layer_ints,
            "extraction_strategy": strategy,
            "extraction_component": component,
            "raw_mode": False,
            "num_pairs": len(enriched_pairs),
            "calibration_norms": calibration_norms,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_kv_heads,
            "pairs": [],
        }
        for i, ep in enumerate(enriched_pairs):
            pair_dict = {
                "prompt": ep.prompt,
                "positive_response": {
                    "model_response": ep.positive_response.model_response,
                    "layers_activations": {},
                },
                "negative_response": {
                    "model_response": ep.negative_response.model_response,
                    "layers_activations": {},
                },
                "label": ep.label,
                "trait_description": ep.trait_description,
            }
            if ep.positive_response.layers_activations:
                for layer_str, act in ep.positive_response.layers_activations.items():
                    if act is not None:
                        pair_dict["positive_response"]["layers_activations"][layer_str] = act.cpu().tolist()
            if ep.negative_response.layers_activations:
                for layer_str, act in ep.negative_response.layers_activations.items():
                    if act is not None:
                        pair_dict["negative_response"]["layers_activations"][layer_str] = act.cpu().tolist()

            qk = qk_data_per_pair[i]
            for side, side_key in [("positive_response", "pos"), ("negative_response", "neg")]:
                side_qk = qk.get(side_key, {})
                q_acts, k_acts = {}, {}
                for key, val in side_qk.items():
                    layer = key[2:]
                    (q_acts if key.startswith("q_") else k_acts)[layer] = val.cpu().tolist()
                pair_dict[side]["q_proj_activations"] = q_acts
                pair_dict[side]["k_proj_activations"] = k_acts
            output_data["pairs"].append(pair_dict)

        out_path = os.path.join(output_dir, f"{task_name}__{strategy}.json")
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=JSON_INDENT)
        written[strategy] = out_path
        if verbose:
            print(f"[multi] wrote {out_path}", flush=True)

    return written
