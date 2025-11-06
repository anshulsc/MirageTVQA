import json
import argparse
from pathlib import Path
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn2, venn3_circles, venn2_circles

def load_top_k_heads_by_accuracy(file_path: Path, k: int) -> tuple:
    """
    Loads the top K heads from a probe result JSON file, sorted by accuracy.

    Args:
        file_path: Path to the JSON file containing a list of ranked heads.
        k: The number of top heads to load.

    Returns:
        A tuple of (set of (layer, head) tuples, dict mapping (layer, head) to accuracy)
    """
    if not file_path.is_file():
        print(f"Error: File not found at {file_path}")
        exit(1)
        
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Sort by accuracy (descending)
    if data and 'accuracy' in data[0]:
        data = sorted(data, key=lambda x: x.get('accuracy', 0), reverse=True)
    else:
        print(f"Warning: No accuracy field found in {file_path}. Using original order.")
    
    # Create a set of (layer, head) tuples for efficient intersection
    head_set = {(item['layer'], item['head']) for item in data[:k]}
    
    # Create dict mapping (layer, head) to accuracy
    head_accuracy = {(item['layer'], item['head']): item.get('accuracy', 0.0) for item in data[:k]}
    
    return head_set, head_accuracy, data[:k]

def get_top_5_by_accuracy(probe_data: dict) -> dict:
    """
    Extract top 5 heads by accuracy from each probe.
    
    Args:
        probe_data: Dict mapping probe names to (head_set, head_accuracy, top_items) tuples
    
    Returns:
        Dict mapping probe names to list of top 5 heads with accuracy
    """
    top_5_dict = {}
    for name, (_, _, top_items) in probe_data.items():
        top_5 = []
        for item in top_items[:5]:
            acc = item.get('accuracy', 'N/A')
            top_5.append({
                'layer': item['layer'],
                'head': item['head'],
                'accuracy': acc
            })
        top_5_dict[name] = top_5
    return top_5_dict

def print_top_5_heads(top_5_dict: dict):
    """Print the top 5 heads by accuracy for each probe."""
    print("\n" + "="*70)
    print("TOP 5 HEADS BY ACCURACY FOR EACH PROBE")
    print("="*70)
    
    for probe_name, top_heads in top_5_dict.items():
        print(f"\n{probe_name}:")
        for i, head in enumerate(top_heads, 1):
            acc = head['accuracy']
            if isinstance(acc, float):
                acc_str = f"{acc:.4f}"
            else:
                acc_str = str(acc)
            print(f"   {i}. Layer {head['layer']:2d}, Head {head['head']:2d} - Accuracy: {acc_str}")

def calculate_average_accuracy(head: tuple, probe_accuracies: dict, probe_names: list) -> float:
    """
    Calculate average accuracy for a head across specified probes.
    
    Args:
        head: (layer, head) tuple
        probe_accuracies: Dict mapping probe names to accuracy dicts
        probe_names: List of probe names to include in average
    
    Returns:
        Average accuracy across probes
    """
    accuracies = []
    for probe_name in probe_names:
        if head in probe_accuracies[probe_name]:
            accuracies.append(probe_accuracies[probe_name][head])
    
    return sum(accuracies) / len(accuracies) if accuracies else 0.0

def analyze_and_report(probe_sets: dict, probe_accuracies: dict, structural_name: str = None):
    """
    Calculates and prints the intersections between different sets of heads.

    Args:
        probe_sets: A dictionary where keys are probe names (str) and
                    values are the corresponding sets of heads.
        probe_accuracies: Dict mapping probe names to accuracy dicts
        structural_name: Name of the structural probe (if any)
    """
    print("\n" + "="*70)
    print("INTERSECTION ANALYSIS REPORT")
    print("="*70)

    probe_names = list(probe_sets.keys())
    
    # Separate language probes
    language_names = [name for name in probe_names if name != structural_name]
    language_sets = {name: probe_sets[name] for name in language_names}

    # --- Analyze language-only intersections first ---
    if len(language_names) >= 2:
        print("\n" + "-"*70)
        print("LANGUAGE PROBES ONLY INTERSECTIONS")
        print("-"*70)
        
        print("\n--- Pairwise Intersections (Language Probes) ---")
        for name1, name2 in combinations(language_names, 2):
            set1 = language_sets[name1]
            set2 = language_sets[name2]
            intersection = set1.intersection(set2)
            
            print(f"\nOverlap between '{name1}' and '{name2}':")
            print(f"   - Number of overlapping heads: {len(intersection)}")
            if intersection:
                # Calculate average accuracy for each head and sort
                heads_with_avg = []
                for head in intersection:
                    avg_acc = calculate_average_accuracy(head, probe_accuracies, [name1, name2])
                    heads_with_avg.append((head, avg_acc))
                
                # Sort by average accuracy (descending)
                heads_with_avg.sort(key=lambda x: x[1], reverse=True)
                
                print(f"   - Overlapping Heads (Layer, Head, Avg Accuracy):")
                for head, avg_acc in heads_with_avg[:10]:
                    print(f"     {head} - Avg Acc: {avg_acc:.4f}")
                if len(heads_with_avg) > 10:
                    print(f"   - ... and {len(heads_with_avg) - 10} more")
        
        # Language-only core intersection
        if len(language_names) >= 2:
            print("\n--- Core Intersection (All Language Probes) ---")
            lang_core = set.intersection(*language_sets.values())
            
            print(f"\nOverlap between all {len(language_names)} language probes:")
            print(f"   - Number of overlapping heads: {len(lang_core)}")
            if lang_core:
                # Calculate average accuracy across all language probes
                heads_with_avg = []
                for head in lang_core:
                    avg_acc = calculate_average_accuracy(head, probe_accuracies, language_names)
                    heads_with_avg.append((head, avg_acc))
                
                # Sort by average accuracy (descending)
                heads_with_avg.sort(key=lambda x: x[1], reverse=True)
                
                print(f"   - Core Language Heads (Layer, Head, Avg Accuracy):")
                for head, avg_acc in heads_with_avg:
                    print(f"     {head} - Avg Acc: {avg_acc:.4f}")

    # --- Analyze all probes including structural ---
    if structural_name:
        print("\n" + "-"*70)
        print("ALL PROBES (INCLUDING STRUCTURAL) INTERSECTIONS")
        print("-"*70)
        
        print("\n--- Pairwise Intersections (Structural with each Language) ---")
        for lang_name in language_names:
            struct_set = probe_sets[structural_name]
            lang_set = probe_sets[lang_name]
            intersection = struct_set.intersection(lang_set)
            
            print(f"\nOverlap between '{structural_name}' and '{lang_name}':")
            print(f"   - Number of overlapping heads: {len(intersection)}")
            if intersection:
                # Calculate average accuracy
                heads_with_avg = []
                for head in intersection:
                    avg_acc = calculate_average_accuracy(head, probe_accuracies, [structural_name, lang_name])
                    heads_with_avg.append((head, avg_acc))
                
                # Sort by average accuracy (descending)
                heads_with_avg.sort(key=lambda x: x[1], reverse=True)
                
                print(f"   - Overlapping Heads (Layer, Head, Avg Accuracy):")
                for head, avg_acc in heads_with_avg:
                    print(f"     {head} - Avg Acc: {avg_acc:.4f}")

        # All probes core intersection
        print("\n--- Core Intersection (All Probes Including Structural) ---")
        all_core = set.intersection(*probe_sets.values())
        
        print(f"\nOverlap between all {len(probe_names)} probes:")
        print(f"   - Number of overlapping heads: {len(all_core)}")
        if all_core:
            # Calculate average accuracy across all probes
            heads_with_avg = []
            for head in all_core:
                avg_acc = calculate_average_accuracy(head, probe_accuracies, probe_names)
                heads_with_avg.append((head, avg_acc))
            
            # Sort by average accuracy (descending)
            heads_with_avg.sort(key=lambda x: x[1], reverse=True)
            
            print(f"   - Core Bottleneck Heads (Layer, Head, Avg Accuracy):")
            for head, avg_acc in heads_with_avg:
                print(f"     {head} - Avg Acc: {avg_acc:.4f}")

    print("\n" + "="*70)

def visualize_intersections(probe_sets: dict, k: int, output_file: Path, structural_name: str):
    """
    Generates and saves Venn diagrams:
    1. Language probes only (excluding Structural)
    2. Language probes + Structural probe
    """
    names = list(probe_sets.keys())
    
    # Separate language probes
    language_names = [name for name in names if name != structural_name]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # --- First diagram: Language probes only ---
    if len(language_names) >= 2:
        plt.sca(axes[0])
        lang_sets = [probe_sets[name] for name in language_names]
        
        if len(language_names) == 2:
            v = venn2(subsets=lang_sets, set_labels=language_names, ax=axes[0])
            title = f'Language Probes Overlap (Top {k} Heads by Accuracy)\n{" vs ".join(language_names)}'
        elif len(language_names) >= 3:
            # Use first 3 for Venn diagram
            v = venn3(subsets=lang_sets[:3], set_labels=language_names[:3], ax=axes[0])
            title = f'Language Probes Overlap (Top {k} Heads by Accuracy)\n{" vs ".join(language_names[:3])}'
            if len(language_names) > 3:
                title += f'\n(Showing first 3 of {len(language_names)} language probes)'
        
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        
        # Style the diagram
        if v:
            for text in v.set_labels:
                if text:
                    text.set_fontsize(11)
            for text in v.subset_labels:
                if text:
                    text.set_fontsize(10)
    else:
        axes[0].text(0.5, 0.5, 'Need at least 2 language probes\nfor visualization', 
                    ha='center', va='center', fontsize=12)
        axes[0].axis('off')
    
    # --- Second diagram: With structural probe ---
    if structural_name and len(language_names) >= 2:
        plt.sca(axes[1])
        
        # Combine with structural
        combined_names = language_names[:2] + [structural_name]
        combined_sets = [probe_sets[name] for name in combined_names]
        
        v = venn3(subsets=combined_sets, set_labels=combined_names, ax=axes[1])
        title = f'Language + Structural Overlap (Top {k} Heads by Accuracy)\n{" vs ".join(combined_names)}'
        axes[1].set_title(title, fontsize=14, fontweight='bold')
        
        # Style the diagram
        if v:
            for text in v.set_labels:
                if text:
                    text.set_fontsize(11)
            for text in v.subset_labels:
                if text:
                    text.set_fontsize(10)
    elif structural_name:
        axes[1].text(0.5, 0.5, 'Need at least 2 language probes\n+ 1 structural probe', 
                    ha='center', va='center', fontsize=12)
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'No structural probe found\n(name should contain "struct")', 
                    ha='center', va='center', fontsize=12)
        axes[1].axis('off')
    
    plt.tight_layout()
    
    # Ensure the output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVenn diagrams saved to: {output_file}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize the intersection of top-performing attention heads from different probe experiments.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--probes',
        nargs='+',
        required=True,
        help="List of probe result files to compare, in the format 'name=path'.\n"
             "Example:\n"
             "  --probes Structural=path/to/structural.json \\\n"
             "           EN-HI=path/to/en_hi.json \\\n"
             "           EN-AR=path/to/en_ar.json"
    )
    parser.add_argument(
        '--k',
        type=int,
        default=100,
        help="Number of top heads to consider from each probe file, ranked by ACCURACY (default: 100)."
    )
    parser.add_argument(
        '--output_plot',
        type=str,
        default="outputs/intersections/head_overlap_venn.png",
        help="Path to save the output Venn diagram plot (default: outputs/intersections/head_overlap_venn.png)."
    )

    args = parser.parse_args()

    # --- Parse the probe arguments into a dictionary ---
    probe_files = {}
    for probe_arg in args.probes:
        if '=' not in probe_arg:
            print(f"Error: Invalid format for --probes argument: '{probe_arg}'. Use 'name=path'.")
            exit(1)
        name, path = probe_arg.split('=', 1)
        probe_files[name] = Path(path)

    # --- Load head sets from files (sorted by accuracy) ---
    print(f"\nLoading top {args.k} heads by accuracy from each probe...")
    probe_data = {
        name: load_top_k_heads_by_accuracy(path, args.k)
        for name, path in probe_files.items()
    }
    
    # Extract sets and accuracies for intersection analysis
    probe_head_sets = {
        name: head_set
        for name, (head_set, _, _) in probe_data.items()
    }
    
    probe_accuracies = {
        name: head_accuracy
        for name, (_, head_accuracy, _) in probe_data.items()
    }
    
    # Identify structural probe
    structural_name = None
    for name in probe_head_sets.keys():
        if 'struct' in name.lower():
            structural_name = name
            break
    
    # Get top 5 heads by accuracy
    top_5_dict = get_top_5_by_accuracy(probe_data)
    
    # --- Print Top 5 Heads ---
    print_top_5_heads(top_5_dict)

    # --- Run Analysis and Visualization ---
    analyze_and_report(probe_head_sets, probe_accuracies, structural_name)
    visualize_intersections(probe_head_sets, args.k, Path(args.output_plot), structural_name)
    
    print("\n" + "="*70)
    print("Analysis Complete.")
    print("="*70)

if __name__ == "__main__":
    main()


"""
USAGE EXAMPLE:
uv run -m src.vis.overlap \
    --k 50 \
    --probes \
        Structural=/data/asca/MirageTVQA/outputs/Qwen2.5-VL-3B-Instruct/structural/results/top_k_heads.json \
        EN-HI=/data/asca/MirageTVQA/outputs/Qwen2.5-VL-3B-Instruct/multilingual_en_hi/results/top_k_heads.json \
        EN-AR=/data/asca/MirageTVQA/outputs/Qwen2.5-VL-3B-Instruct/multilingual_en_ar/results/top_k_heads.json \
        EN-ZH=/data/asca/MirageTVQA/outputs/Qwen2.5-VL-3B-Instruct/multilingual_en_zh_cn/results/top_k_heads.json \
        EN-RU=/data/asca/MirageTVQA/outputs/Qwen2.5-VL-3B-Instruct/multilingual_en_ru_formal/results/top_k_heads.json \
    --output_plot outputs/intersections/struct_qwen3b_overlap.png
"""