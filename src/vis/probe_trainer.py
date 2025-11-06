import json
import joblib
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import logging
import warnings
from sklearn.exceptions import ConvergenceWarning



class ProbeTrainer:
    def __init__(self, config):
        self.config = config
        self.features_dir = config.FEATURES_DIR
        self.probe_dir = config.PROBE_DIR
        self.results_dir = config.RESULTS_DIR
        self.logger = logging.getLogger(__name__)

    def load_data_for_head(self, layer_idx, head_idx):
        filepath = self.features_dir / f"features_layer_{layer_idx}_head_{head_idx}.jsonl"
        features, labels = [], []
        with open(filepath, "r") as f:
            for line in f:
                data = json.loads(line)
                features.append(data["feature"])
                labels.append(data["label"])
        return np.array(features), np.array(labels)

    def train_all_probes(self):
        self.logger.info("Starting probe training for all attention heads...")
        
        results = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            for l in tqdm(range(self.config.NUM_LAYERS), desc="Training Probes (Layers)"):
                for h in range(self.config.NUM_HEADS):
                    try:
                        X, y = self.load_data_for_head(l, h)
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, 
                            test_size=1 - self.config.TRAIN_TEST_SPLIT, 
                            random_state=self.config.RANDOM_SEED,
                            stratify=y
                        )

                        probe = make_pipeline(
                            StandardScaler(), 
                            LinearSVC(
                                random_state=self.config.RANDOM_SEED, 
                                dual="auto",  
                                max_iter=2000 
                            )
                        )
                        probe.fit(X_train, y_train)
                        
                        accuracy = probe.score(X_test, y_test)
                        results.append({"layer": l, "head": h, "accuracy": accuracy})
                        
                        probe_path = self.probe_dir / f"probe_layer_{l}_head_{h}.joblib"
                        joblib.dump(probe, probe_path)

                    except Exception as e:
                        self.logger.error(f"Failed to train probe for layer {l}, head {h}: {e}")

        results_path = self.results_dir / "probe_accuracies.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Probe training complete. Accuracies saved to {results_path}")