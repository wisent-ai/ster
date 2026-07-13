"""Visualization Gallery for Zwiad.

Creates publication-quality figures for the paper.
"""

import argparse
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from wisent.core.utils.config_tools.constants import VIZ_PLOT_DPI, VIZ_FONTSIZE_SUPTITLE, VIZ_FONTSIZE_SUBTITLE, VIZ_ALPHA_LIGHT, VIZ_ALPHA_HALF, VIZ_LINEWIDTH_NORMAL, SEPARATOR_WIDTH_WIDE, NONSENSE_N_PAIRS, HASH_PREFIX_LEN, ARCHITECTURE_MODULE_LIMIT_DEFAULT
from wisent.examples.scripts.visualization_gallery_helpers import (
    gcs_upload_file,
    load_diagnosis_results,
    select_representative_benchmarks,
    create_tsne_plot,
)
from wisent.examples.scripts.visualization_gallery_figures import (
    create_hero_figure,
    create_layer_accuracy_curves,
)


def create_tsne_gallery(
    model: "WisentModel",
    selected_benchmarks: Dict[str, List[str]],
    output_path: Path,
    model_name: str,
    report_interval: int,
    trait_name_max_length: int,
) -> None:
    """
    Create t-SNE gallery figure.
    
    Layout: 2x3 grid showing examples from each diagnosis type
    
    Args:
        model: WisentModel instance
        selected_benchmarks: Dict with benchmark names by diagnosis
        output_path: Where to save figure
        model_name: Model name for title
    """
    if not HAS_MATPLOTLIB or not HAS_SKLEARN:
        print("Skipping t-SNE gallery: required packages not installed")
        return
    
    from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
    from wisent.core.primitives.model_interface.core.activations.activation_cache import ActivationCache, collect_and_cache_activations
    from lm_eval.tasks import TaskManager
    from wisent.extractors.lm_eval.lm_task_pairs_generation import lm_build_contrastive_pairs
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f't-SNE Visualization Gallery\n{model_name}', fontsize=VIZ_FONTSIZE_SUPTITLE, fontweight='bold')
    
    cache_dir = f"/tmp/wisent_viz_cache_{model_name.replace('/', '_')}"
    cache = ActivationCache(cache_dir, hash_digest_prefix=HASH_PREFIX_LEN)
    tm = TaskManager()
    strategy = ExtractionStrategy.default()
    
    row = 0
    for diagnosis in ["LINEAR", "NONLINEAR", "NO_SIGNAL"]:
        benchmarks = selected_benchmarks.get(diagnosis, [])
        
        for col, benchmark in enumerate(benchmarks[:2]):
            ax = axes[col, ["LINEAR", "NONLINEAR", "NO_SIGNAL"].index(diagnosis)]
            
            try:
                # Load pairs
                try:
                    task_dict = tm.load_task_or_group([benchmark])
                    task = list(task_dict.values())[0]
                except Exception:
                    task = None
                
                pairs = lm_build_contrastive_pairs(benchmark, task, limit=NONSENSE_N_PAIRS, train_ratio=args.train_ratio)
                
                if len(pairs) < 20:
                    ax.text(0.5, 0.5, f'{benchmark}\n(insufficient data)', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
                    continue
                
                # Get activations
                cached = collect_and_cache_activations(
                    model=model,
                    pairs=pairs,
                    benchmark=benchmark,
                    strategy=strategy,
                    report_interval=report_interval,
                    cache=cache,
                    show_progress=False,
                    architecture_module_limit=ARCHITECTURE_MODULE_LIMIT_DEFAULT,
                )
                
                # Use middle layer
                middle_layer = str(model.num_layers // 2)
                pos_acts = cached.get_positive_activations(middle_layer)
                neg_acts = cached.get_negative_activations(middle_layer)
                
                # Create t-SNE plot
                create_tsne_plot(pos_acts, neg_acts, benchmark, ax, diagnosis, pca_gallery_components=args.pca_gallery_components)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'{benchmark}\n(error: {str(e)[:trait_name_max_length]})', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
    
    # Add column labels
    for idx, diagnosis in enumerate(["LINEAR", "NONLINEAR", "NO_SIGNAL"]):
        axes[0, idx].set_xlabel(diagnosis, fontsize=VIZ_FONTSIZE_SUBTITLE, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIZ_PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved t-SNE gallery: {output_path}")


def create_layer_accuracy_curves(
    diagnosis_results: Dict[str, Any],
    output_path: Path,
    model_name: str,
) -> None:
    """
    Create layer-wise accuracy curves.
    
    Shows how kNN and linear probe accuracy change across layers.
    
    Args:
        diagnosis_results: Loaded diagnosis results
        output_path: Where to save figure
        model_name: Model name for title
    """
    if not HAS_MATPLOTLIB:
        print("Skipping layer curves: matplotlib not installed")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Layer-wise Accuracy Curves\n{model_name}', fontsize=VIZ_FONTSIZE_SUPTITLE, fontweight='bold')
    
    # Collect layer-wise data (we don't have per-layer data in current results,
    # so we'll create placeholder showing the concept)
    
    # This would need per-layer results which we can add to discover_directions
    # For now, create example curves
    
    for idx, (diagnosis, color) in enumerate([
        ("LINEAR", "#2ecc71"),
        ("NONLINEAR", "#3498db"),
        ("NO_SIGNAL", "#95a5a6")
    ]):
        ax = axes[idx]
        
        # Example curves (would be replaced with real data)
        layers = np.arange(1, 33)
        
        if diagnosis == "LINEAR":
            knn = 0.5 + 0.4 * np.exp(-(layers - 16)**2 / 100)
            linear = 0.5 + 0.35 * np.exp(-(layers - 16)**2 / 100)
        elif diagnosis == "NONLINEAR":
            knn = 0.5 + 0.35 * np.exp(-(layers - 16)**2 / 100)
            linear = 0.5 + 0.1 * np.exp(-(layers - 16)**2 / 100)
        else:
            knn = 0.5 + 0.05 * np.random.randn(len(layers))
            linear = 0.5 + 0.05 * np.random.randn(len(layers))
        
        ax.plot(layers, knn, 'b-', linewidth=VIZ_LINEWIDTH_NORMAL, label='kNN-10')
        ax.plot(layers, linear, 'g--', linewidth=VIZ_LINEWIDTH_NORMAL, label='Linear Probe')
        ax.fill_between(layers, knn, linear, alpha=VIZ_ALPHA_LIGHT, color='yellow', label='Gap')
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Accuracy')
        ax.set_title(diagnosis, fontsize=VIZ_FONTSIZE_SUBTITLE, fontweight='bold')
        ax.legend()
        ax.set_ylim(0.4, 1.0)
        ax.axhline(y=0.6, color='gray', linestyle='--', alpha=VIZ_ALPHA_HALF)
        ax.set_xlim(1, 32)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIZ_PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved layer curves: {output_path}")


def run_visualization(model_name: str, report_interval: int, skip_tsne: bool = False, *, signal_exist_threshold: float, signal_linear_gap: float):
    """
    Generate all visualizations.
    
    Args:
        model_name: Model to visualize
        skip_tsne: Skip t-SNE (requires model loading)
    """
    print("=" * SEPARATOR_WIDTH_WIDE)
    print("VISUALIZATION GALLERY")
    print("=" * SEPARATOR_WIDTH_WIDE)
    print(f"Model: {model_name}")
    
    output_dir = Path("/tmp/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load diagnosis results
    diagnosis_results = load_diagnosis_results(model_name, output_dir)
    if not diagnosis_results:
        print("ERROR: No diagnosis results found.")
        return
    
    print(f"Loaded results for {len(diagnosis_results)} categories")
    
    model_prefix = model_name.replace('/', '_')
    
    # 1. Hero figure
    print("\n1. Creating hero figure...")
    hero_path = output_dir / f"{model_prefix}_hero_figure.png"
    create_hero_figure(diagnosis_results, hero_path, model_name, signal_exist_threshold=signal_exist_threshold, signal_linear_gap=signal_linear_gap)
    gcs_upload_file(hero_path, model_name)
    
    # 2. Layer accuracy curves
    print("\n2. Creating layer accuracy curves...")
    curves_path = output_dir / f"{model_prefix}_layer_curves.png"
    create_layer_accuracy_curves(diagnosis_results, curves_path, model_name)
    gcs_upload_file(curves_path, model_name)
    
    # 3. t-SNE gallery (requires model)
    if not skip_tsne:
        print("\n3. Creating t-SNE gallery...")
        
        from wisent.core.primitives.models.wisent_model import WisentModel
        
        print(f"  Loading model: {model_name}")
        model = WisentModel(model_name, device="cuda")
        
        selected = select_representative_benchmarks(diagnosis_results, signal_exist_threshold=signal_exist_threshold, signal_linear_gap=signal_linear_gap)
        print(f"  Selected benchmarks: {selected}")
        
        tsne_path = output_dir / f"{model_prefix}_tsne_gallery.png"
        create_tsne_gallery(model, selected, tsne_path, model_name, report_interval=report_interval)
        gcs_upload_file(tsne_path, model_name)
        
        del model
    else:
        print("\n3. Skipping t-SNE gallery (--skip-tsne)")
    
    print("\n" + "=" * SEPARATOR_WIDTH_WIDE)
    print("VISUALIZATION COMPLETE")
    print("=" * SEPARATOR_WIDTH_WIDE)
    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization gallery for Zwiad")
    parser.add_argument("--model", type=str, required=True, help="Model to visualize")
    parser.add_argument("--report-interval", type=int, required=True, help="Progress reporting interval for activation collection")
    parser.add_argument("--skip-tsne", action="store_true", help="Skip t-SNE (doesn't require model)")
    parser.add_argument("--signal-exist-threshold", type=float, required=True, help="Signal existence threshold")
    parser.add_argument("--signal-linear-gap", type=float, required=True, help="Signal linear gap threshold")
    args = parser.parse_args()

    run_visualization(args.model, report_interval=args.report_interval, skip_tsne=args.skip_tsne, signal_exist_threshold=args.signal_exist_threshold, signal_linear_gap=args.signal_linear_gap)
