from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import torch
from torch.nn import Module
import matplotlib.pyplot as plt
from analysers.base import AbstractAnalyser


class EffectiveRankAnalyser(AbstractAnalyser):
    """
    Analyser for computing and tracking effective rank of activation distributions.
    Effective rank measures the dimensionality of data matrices and is useful for
    understanding how much of the representational capacity is being used.

    Reference: Roy & Vetterli (2007), "The Effective Rank: A measure of effective dimensionality"
    """

    def __init__(self, tmp_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tmp_path = Path(tmp_path)
        self.points = {}

    @staticmethod
    def compute_effective_rank(activations: torch.Tensor) -> float:
        """
        Compute effective rank of activation tensor.

        Args:
            activations: Tensor of shape (batch, seq_len, features) or (batch, features)

        Returns:
            Effective rank as a float between 0 and min(dimensions)
        """
        with torch.no_grad():
            activations = activations.detach().cpu().float()

            if activations.dim() == 3:
                b, n, d = activations.shape
                activations = activations.reshape(-1, d)
            elif activations.dim() == 2:
                pass
            else:
                activations = activations.flatten(-2, -1) if activations.dim() > 2 else activations

            if activations.shape[0] < activations.shape[1]:
                activations = activations.T

            try:
                U, S, Vh = torch.linalg.svd(activations, full_matrices=False)
                singular_values = S.numpy()

                singular_values = singular_values[singular_values > 1e-6]
                if len(singular_values) == 0:
                    return 0.0

                total_sum = singular_values.sum()
                if total_sum == 0:
                    return 0.0

                p = singular_values / total_sum
                p = p[p > 0]

                entropy = -np.sum(p * np.log(p))
                effective_rank = np.exp(entropy)

                max_rank = min(activations.shape)
                effective_rank = min(effective_rank, max_rank)

                return float(effective_rank)
            except Exception:
                return 0.0

    def add_analytical_point(self, step: int, model: Module, timestep: int = None, *args, **kwargs):
        """
        Collect effective rank for all layers with activations at given step.

        Args:
            step: Training step number
            model: The neural network model
            timestep: Optional diffusion timestep for context
        """
        layers_data = {}
        layer_idx = 1

        for name, module in model.named_modules():
            in_activation = getattr(module, 'in_activation', None)
            if in_activation is not None:
                erank = self.compute_effective_rank(in_activation)
                layers_data[layer_idx] = {
                    'name': name,
                    'effective_rank': erank,
                }
                layer_idx += 1

        if layers_data:
            if step not in self.points:
                self.points[step] = {}
            self.points[step]['timestep' if timestep is not None else 'layers'] = layers_data

    def make_report(self, save_path: Optional[Path] = None, *args, **kwargs):
        """
        Generate comprehensive effective rank analysis report.
        """
        if not self.points:
            return

        steps = sorted(self.points.keys())
        self.tmp_path.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        all_layers = []
        for step in steps:
            if 'layers' in self.points[step]:
                for layer_idx, data in self.points[step]['layers'].items():
                    erank = data['effective_rank']
                    if layer_idx not in [l[0] for l in all_layers]:
                        all_layers.append((layer_idx, []))
                    for i, (idx, _) in enumerate(all_layers):
                        if idx == layer_idx:
                            all_layers[i][1].append(erank)

        if all_layers:
            ax = axes[0]
            for layer_idx, eranks in all_layers:
                ax.plot(steps, eranks[:len(steps)], marker='o', label=f'Layer {layer_idx}', alpha=0.7)
            ax.set_xlabel('Training Step', fontsize=12)
            ax.set_ylabel('Effective Rank', fontsize=12)
            ax.set_title('Effective Rank Over Training', fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            final_step = steps[-1] if steps else 0
            if final_step in self.points and 'layers' in self.points[final_step]:
                layer_data = self.points[final_step]['layers']
                layer_indices = sorted(layer_data.keys())
                eranks = [layer_data[idx]['effective_rank'] for idx in layer_indices]
                colors = plt.cm.viridis(np.linspace(0, 1, len(layer_indices)))
                ax.bar(layer_indices, eranks, color=colors, alpha=0.8)
                ax.set_xlabel('Layer Index', fontsize=12)
                ax.set_ylabel('Effective Rank', fontsize=12)
                ax.set_title(f'Final Effective Rank Distribution (Step {final_step})', fontsize=14)
                ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_dir = save_path if save_path else self.tmp_path
        save_dir.mkdir(parents=True, exist_ok=True)
        out_file = save_dir / 'effective_rank_report.png'
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()

        self._save_numerical_report(save_dir)

    def _save_numerical_report(self, save_dir: Path):
        """Save effective rank values as text report."""
        save_dir.mkdir(parents=True, exist_ok=True)
        report_path = save_dir / 'effective_rank_values.txt'

        with open(report_path, 'w') as f:
            f.write("Effective Rank Analysis Report\n")
            f.write("=" * 60 + "\n\n")

            steps = sorted(self.points.keys())
            for step in steps:
                f.write(f"Step {step}:\n")
                if 'layers' in self.points[step]:
                    layer_data = self.points[step]['layers']
                    for layer_idx in sorted(layer_data.keys()):
                        data = layer_data[layer_idx]
                        f.write(f"  Layer {layer_idx} ({data['name']}): {data['effective_rank']:.4f}\n")
                f.write("\n")
