"""Build enriched pairs JSON from database or by generation + activation collection."""
import json
import math
import os
import struct
from collections import defaultdict
from typing import Optional, Sequence

from wisent.core.utils.config_tools.constants import BYTES_PER_MB
from wisent.core.control.steering_methods.configs.optimal import get_optimal_extraction_strategy, get_optimal


def build_enriched_from_db(
    model_name: str,
    task_name: str,
    work_dir: str,
    extraction_strategy: str,
    limit: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    num_key_value_heads: Optional[int] = None,
    database_url: Optional[str] = None,
) -> Optional[str]:
    """
    Build enriched pairs JSON from Supabase database.
    Returns path to the enriched file, or None if DB data unavailable.
    """
    try:
        import psycopg2
    except ImportError:
        print("  DB: psycopg2 not available, skipping DB lookup")
        return None

    db_url = database_url or os.environ.get("DATABASE_URL")
    if not db_url:
        print("  DB: No DATABASE_URL set, skipping DB lookup")
        return None
    if "sslmode=" not in db_url:
        db_url += "?sslmode=require" if "?" not in db_url else "&sslmode=require"

    try:
        conn = psycopg2.connect(db_url)
    except Exception as e:
        print(f"  DB: Connection failed: {e}")
        return None

    cur = conn.cursor()
    try:
        return _query_and_build(
            cur, model_name, task_name, work_dir,
            extraction_strategy, limit,
            num_attention_heads, num_key_value_heads)
    except Exception as e:
        print(f"  DB lookup failed: {e}")
        return None
    finally:
        cur.close()
        conn.close()


def _query_and_build(
    cur, model_name, task_name, work_dir,
    extraction_strategy, limit,
    num_attention_heads, num_key_value_heads,
):
    """Run DB queries and build enriched JSON. Returns file path or None."""
    cur.execute('SELECT id FROM "Model" WHERE "huggingFaceId" = %s', (model_name,))
    row = cur.fetchone()
    if not row:
        print(f"  DB: Model {model_name} not found")
        return None
    model_id = row[0]

    cur.execute(
        'SELECT id, name FROM "ContrastivePairSet" WHERE name = %s OR name LIKE %s ORDER BY name LIMIT 1',
        (task_name, '%/' + task_name))
    row = cur.fetchone()
    if not row:
        print(f"  DB: Task {task_name} not found")
        return None
    set_id = row[0]
    print(f"  DB: Matched task '{task_name}' → '{row[1]}' (id={set_id})")

    cur.execute('''
        SELECT id, "positiveExample", "negativeExample"
        FROM "ContrastivePair" WHERE "setId" = %s ORDER BY id
    ''', (set_id,))
    pair_rows = cur.fetchall()
    if not pair_rows:
        print(f"  DB: No pairs for {task_name}")
        return None

    pairs_text = {}
    for pid, pos_ex, neg_ex in pair_rows:
        if "\n\n" in pos_ex:
            prompt = pos_ex.rsplit("\n\n", 1)[0]
            pos_resp = pos_ex.rsplit("\n\n", 1)[1]
        else:
            prompt, pos_resp = pos_ex, ""
        neg_resp = neg_ex.rsplit("\n\n", 1)[1] if "\n\n" in neg_ex else neg_ex
        pairs_text[pid] = {"prompt": prompt, "positive": pos_resp, "negative": neg_resp}

    cur.execute('''
        SELECT DISTINCT layer FROM "Activation"
        WHERE "modelId" = %s AND "contrastivePairSetId" = %s AND "extractionStrategy" = %s
        ORDER BY layer
    ''', (model_id, set_id, extraction_strategy))
    layers = [r[0] for r in cur.fetchall()]
    if not layers:
        print(f"  DB: No activations for {model_name}/{task_name}/{extraction_strategy}")
        return None

    cur.execute('''
        SELECT "contrastivePairId", layer, "activationData", "isPositive"
        FROM "Activation"
        WHERE "modelId" = %s AND "contrastivePairSetId" = %s AND "extractionStrategy" = %s
        ORDER BY "contrastivePairId", layer, "isPositive"
    ''', (model_id, set_id, extraction_strategy))

    act_data = defaultdict(lambda: defaultdict(dict))
    for pid, layer, blob, is_pos in cur.fetchall():
        n = len(blob) // 4
        act_data[pid][layer]["pos" if is_pos else "neg"] = list(struct.unpack(f'{n}f', blob))

    complete_pids = []
    for pid in sorted(pairs_text.keys()):
        if pid not in act_data:
            continue
        has_all = all(
            "pos" in act_data[pid].get(l, {}) and "neg" in act_data[pid].get(l, {})
            for l in layers)
        if has_all:
            complete_pids.append(pid)

    if not complete_pids:
        print(f"  DB: No complete pairs (all layers + both sides) for {task_name}")
        return None
    if limit and len(complete_pids) > limit:
        complete_pids = complete_pids[:limit]

    print(f"  DB: Found {len(complete_pids)} complete pairs across {len(layers)} layers")

    calibration_norms = {}
    for layer in layers:
        norms = []
        for pid in complete_pids:
            for side in ("pos", "neg"):
                vec = act_data[pid][layer][side]
                norms.append(math.sqrt(sum(x * x for x in vec)))
        calibration_norms[str(layer)] = sum(norms) / len(norms) if norms else 0.0

    enriched_pairs = _assemble_pairs(complete_pids, pairs_text, act_data, layers, task_name)

    output = {
        "task_name": task_name, "trait_label": task_name,
        "model": model_name, "layers": layers,
        "extraction_strategy": extraction_strategy,
        "extraction_component": get_optimal("extraction_component"),
        "raw_mode": False, "num_pairs": len(enriched_pairs),
        "calibration_norms": calibration_norms,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "pairs": enriched_pairs,
    }

    out_path = os.path.join(work_dir, "enriched_from_db.json")
    with open(out_path, 'w') as f:
        json.dump(output, f)

    size_mb = os.path.getsize(out_path) / BYTES_PER_MB
    print(f"  DB: Wrote enriched file ({size_mb:.1f} MB): {out_path}")
    return out_path


def _assemble_pairs(complete_pids, pairs_text, act_data, layers, task_name):
    """Assemble enriched pair dicts from text and activation data."""
    enriched = []
    for pid in complete_pids:
        pt = pairs_text[pid]
        pair_dict = {
            "prompt": pt["prompt"],
            "positive_response": {
                "model_response": pt["positive"],
                "layers_activations": {},
                "q_proj_activations": {}, "k_proj_activations": {},
            },
            "negative_response": {
                "model_response": pt["negative"],
                "layers_activations": {},
                "q_proj_activations": {}, "k_proj_activations": {},
            },
            "label": task_name, "trait_description": "",
        }
        for layer in layers:
            ls = str(layer)
            pair_dict["positive_response"]["layers_activations"][ls] = act_data[pid][layer]["pos"]
            pair_dict["negative_response"]["layers_activations"][ls] = act_data[pid][layer]["neg"]
        enriched.append(pair_dict)
    return enriched


def build_enriched_from_hf(
    model_name: str, task_name: str, layer: int,
    extraction_strategy: str, work_dir: str,
    train_pairs_file: Optional[str] = None, limit: Optional[int] = None,
) -> Optional[str]:
    """Build enriched pairs JSON from HuggingFace cached activations.
    Returns path to the enriched file, or None if HF data unavailable.
    """
    try:
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
            load_activations_from_hf, load_pair_texts_from_hf,
        )
    except ImportError:
        print("  HF: Required packages not available, skipping")
        return None
    try:
        if train_pairs_file and os.path.exists(train_pairs_file):
            with open(train_pairs_file) as f:
                data = json.load(f)
            pairs_text = {}
            for i, p in enumerate(data.get("pairs", [])):
                pos = p.get("positive_response", {})
                neg = p.get("negative_response", {})
                pairs_text[i] = {
                    "prompt": p.get("prompt", ""),
                    "positive": pos.get("model_response", "") if isinstance(pos, dict) else str(pos),
                    "negative": neg.get("model_response", "") if isinstance(neg, dict) else str(neg),
                }
        else:
            pairs_text = load_pair_texts_from_hf(task_name, limit)
        if not pairs_text:
            print(f"  HF: No pair texts for {task_name}")
            return None
        n_pairs = min(len(pairs_text), limit) if limit else len(pairs_text)
        pos_act, neg_act = load_activations_from_hf(
            model_name, task_name, layer, extraction_strategy, limit=n_pairs)
        if not len(pos_act):
            print(f"  HF: No activations for {model_name}/{task_name}/layer {layer}")
            return None
        n_usable = min(len(pairs_text), len(pos_act))
        sorted_pids = sorted(pairs_text.keys())[:n_usable]
        act_data = defaultdict(lambda: defaultdict(dict))
        for i, pid in enumerate(sorted_pids):
            act_data[pid][layer]["pos"] = pos_act[i].tolist()
            act_data[pid][layer]["neg"] = neg_act[i].tolist()
        cal = [math.sqrt(sum(x * x for x in act_data[p][layer][s]))
               for p in sorted_pids for s in ("pos", "neg")]
        calibration_norms = {str(layer): sum(cal) / len(cal)}
        enriched_pairs = _assemble_pairs(sorted_pids, pairs_text, act_data, [layer], task_name)
        output = {
            "task_name": task_name, "trait_label": task_name, "model": model_name,
            "layers": [layer], "extraction_strategy": extraction_strategy,
            "extraction_component": get_optimal("extraction_component"), "raw_mode": False,
            "num_pairs": len(enriched_pairs), "calibration_norms": calibration_norms,
            "pairs": enriched_pairs,
        }
        out_path = os.path.join(work_dir, "enriched_from_hf.json")
        with open(out_path, 'w') as f:
            json.dump(output, f)
        size_mb = os.path.getsize(out_path) / BYTES_PER_MB
        print(f"  HF: Wrote enriched ({size_mb:.1f} MB, {len(enriched_pairs)} pairs, layer {layer})")
        return out_path
    except Exception as e:
        print(f"  HF: Failed: {e}")
        return None


def build_enriched_from_hf_strict(
    model_name: str,
    task_name: str,
    layer: int,
    extraction_strategy: str,
    work_dir: str,
    expected_pair_ids: Sequence[int],
    completion_manifest_file: Optional[str] = None,
) -> str:
    """Build enriched JSON from one exact, manifest-proven HF activation shard.

    Unlike :func:`build_enriched_from_hf`, this API never uses caches, limits,
    generated pairs, positional joins, or alternate extraction formats. The
    output pair order is exactly ``expected_pair_ids``. Validation failures are
    machine-readable ``ValueError`` instances produced by the strict loaders.
    """
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
        _strict_value_error,
        load_activations_from_hf_strict,
        load_pair_texts_from_hf_strict,
    )

    pos_activations, neg_activations, ordered_pair_ids = load_activations_from_hf_strict(
        model_name=model_name,
        task_name=task_name,
        layer=layer,
        extraction_strategy=extraction_strategy,
        expected_pair_ids=expected_pair_ids,
        completion_manifest_file=completion_manifest_file,
    )
    pair_texts = load_pair_texts_from_hf_strict(task_name, ordered_pair_ids)
    if pos_activations.shape[0] != len(ordered_pair_ids) or neg_activations.shape[0] != len(ordered_pair_ids):
        raise _strict_value_error(
            "enriched_support_mismatch",
            "Activation tensors do not cover the exact requested pair support",
            expected_count=len(ordered_pair_ids),
            positive_count=pos_activations.shape[0],
            negative_count=neg_activations.shape[0],
        )

    layer_key = str(layer)
    enriched_pairs = []
    calibration_norm_sum = 0.0
    for index, pair_id in enumerate(ordered_pair_ids):
        pair_text = pair_texts[pair_id]
        positive = pos_activations[index].tolist()
        negative = neg_activations[index].tolist()
        calibration_norm_sum += math.sqrt(sum(value * value for value in positive))
        calibration_norm_sum += math.sqrt(sum(value * value for value in negative))
        enriched_pairs.append({
            "pair_id": pair_id,
            "prompt": pair_text["prompt"],
            "positive_response": {
                "model_response": pair_text["positive"],
                "layers_activations": {layer_key: positive},
                "q_proj_activations": {},
                "k_proj_activations": {},
            },
            "negative_response": {
                "model_response": pair_text["negative"],
                "layers_activations": {layer_key: negative},
                "q_proj_activations": {},
                "k_proj_activations": {},
            },
            "label": task_name,
            "trait_description": "",
        })

    output = {
        "task_name": task_name,
        "trait_label": task_name,
        "model": model_name,
        "layers": [layer],
        "extraction_strategy": extraction_strategy,
        "extraction_component": "residual_stream",
        "raw_mode": False,
        "num_pairs": len(enriched_pairs),
        "pair_ids": ordered_pair_ids,
        "calibration_norms": {
            layer_key: calibration_norm_sum / (2 * len(ordered_pair_ids)),
        },
        "pairs": enriched_pairs,
    }
    out_path = os.path.join(work_dir, "enriched_from_hf_strict.json")
    try:
        with open(out_path, "w") as handle:
            json.dump(output, handle)
    except OSError as exc:
        raise _strict_value_error(
            "enriched_output_write_failed", "Could not write strict enriched JSON",
            path=out_path, error=str(exc),
        ) from exc
    return out_path


def generate_and_collect_enriched(
    model_name, benchmark, work_dir, limit, device, cached_model=None,
):
    """Generate contrastive pairs and collect activations from model."""
    from wisent.core.utils.cli.optimize_steering.data.contrastive_pairs_data import (
        execute_generate_pairs_from_task,
    )
    from wisent.core.utils.cli.optimize_steering.data.activations_data import (
        execute_get_activations,
    )
    import argparse

    pairs_file = os.path.join(work_dir, "pairs.json")
    enriched_file = os.path.join(work_dir, "enriched_all_layers.json")

    print(f"  Generating contrastive pairs for {benchmark}...")
    execute_generate_pairs_from_task(argparse.Namespace(
        task_name=benchmark, output=pairs_file,
        limit=limit, verbose=False))

    num_layers = cached_model.num_layers if cached_model else 16
    print(f"  Collecting activations (all {num_layers} layers)...")
    execute_get_activations(argparse.Namespace(
        pairs_file=pairs_file, model=model_name,
        output=enriched_file, layers=None,
        extraction_strategy=get_optimal_extraction_strategy(), device=device,
        verbose=False, timing=False, raw=False,
        cached_model=cached_model))
    return enriched_file
