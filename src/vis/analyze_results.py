import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class ResultAnalyzer:
    def __init__(self, config):
        self.config = config
        self.results_dir = config.RESULTS_DIR
        self.visualization_dir = config.VISUALIZATION_DIR
        self.logger = logging.getLogger(__name__)

    def analyze(self):
        self.logger.info("Analyzing probe results...")
        
        results_path = self.results_dir / "probe_accuracies.json"
        with open(results_path, "r") as f:
            results = json.load(f)

        if not results:
            self.logger.error("No probe results found to analyze.")
            return

        # 2. Identify Top-K language-specific heads
        sorted_heads = sorted(results, key=lambda x: x["accuracy"], reverse=True)
        top_k_heads = sorted_heads[:self.config.TOP_K]
        
        self.logger.info(f"Top {min(10, self.config.TOP_K)} Language-Specific Heads:")
        for i, head_info in enumerate(top_k_heads[:10]):
            self.logger.info(
                f"{i+1}. Layer {head_info['layer']}, Head {head_info['head']} "
                f"-> Accuracy: {head_info['accuracy']:.4f}"
            )
            
        # Save the list of top-k heads
        top_k_path = self.results_dir / "top_k_heads.json"
        with open(top_k_path, "w") as f:
            json.dump(top_k_heads, f, indent=2)
        self.logger.info(f"List of top {self.config.TOP_K} heads saved to {top_k_path}")

        # 3. Create Heatmap Visualization (like Figure 7b)
        accuracy_matrix = np.zeros((self.config.NUM_LAYERS, self.config.NUM_HEADS))
        for res in results:
            accuracy_matrix[res['layer'], res['head']] = res['accuracy']
            
        plt.figure(figsize=(12, 10))
        sns.heatmap(accuracy_matrix, cmap="viridis", vmin=0.5) # vmin=0.5 for chance level
        # plt.title(f"Probe Accuracy per Attention Head ({self.config.} vs. {self.config.LANG_B})")
        plt.xlabel("Head Index")
        plt.ylabel("Layer Index")
        
        heatmap_path = self.visualization_dir / "probe_accuracy_heatmap.png"
        plt.savefig(heatmap_path)
        plt.close()
        
        self.logger.info(f"Accuracy heatmap saved to {heatmap_path}")