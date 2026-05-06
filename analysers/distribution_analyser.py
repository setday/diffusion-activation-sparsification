from pathlib import Path
from typing import Optional


import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torch.nn import Module

from analysers.base import AbstractAnalyser


class DistributionAnalyser(AbstractAnalyser):
    """
    Analyser for comparing the weights of two models to identify any drift that may have occurred during training or fine-tuning. This analyser can be used to assess the stability of a model's weights and to identify any significant changes that may indicate overfitting or other issues.
    """
    def __init__(self, tmp_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tmp_path = tmp_path
        self.points = {}

    def add_analytical_point(self, step: int, model: Module, *args, **kwargs):
        """
        Add an analytical point to the analyser. This method should be implemented by subclasses to define how analytical points are added and what information they contain.
        """
        
        layers_data = {}
        layer_idx = 1
        # for each activation layer in the model, save the distribution of pre-activation as plot to the tmp_path with the name of the layer and the step
        for name, module in model.named_modules():
            # Assuming the module has an attribute 'pre_activation_values' that stores the pre-activation values
            pre_activation_values = getattr(module, 'in_activation', None)
            if pre_activation_values is not None:
                # Calculate histogram to save memory
                with torch.no_grad():
                    # Flatten the tensor
                    vals = pre_activation_values.detach().cpu().float().flatten()
                    # Calculate histogram between -15 and 15
                    hist, bin_edges = np.histogram(vals.numpy(), bins=100, range=(-15, 15), density=True)
                    # Convert density to ratio (%)
                    hist = hist * (bin_edges[1] - bin_edges[0]) * 100
                    
                    layers_data[layer_idx] = {
                        'hist': hist,
                        'bins': bin_edges[:-1] + np.diff(bin_edges)/2  # bin centers
                    }
                layer_idx += 1

        self.tmp_path.mkdir(parents=True, exist_ok=True)
        
        if not layers_data:
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Setup colormap (copper-like)
        cmap = plt.get_cmap('Oranges')
        num_layers = max(layers_data.keys())
        norm = mpl.colors.Normalize(vmin=1, vmax=num_layers)
        
        for idx, data in layers_data.items():
            color = cmap(norm(idx))
            ax.plot(data['bins'], data['hist'], color=color, alpha=0.8, linewidth=1.5)
            
        ax.set_xlim(-15, 15)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Pre-Activation Value', fontsize=12)
        ax.set_ylabel('Ratio (%)', fontsize=12)
        ax.grid(True, linestyle='-', alpha=0.7)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, ticks=[1, max(1, num_layers//4), max(1, num_layers//2), max(1, 3*num_layers//4), num_layers])
        cbar.set_label('Layer')
        
        plt.tight_layout()
        
        out_file = self.tmp_path / f'distribution_step_{step}.png'
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()

        # Optionally store for make_report if needed later
        self.points[step] = layers_data

    def make_report(self, save_path: Optional[Path] = None, *args, **kwargs):
        """
        Generate a report based on the analytical points collected by the analyser. This method should be implemented by subclasses to define how the report is generated and what information it contains.
        """        
        steps = sorted(self.points.keys())
        if not steps:
            return
            
        N = len(steps)
        fig, axes = plt.subplots(1, N, figsize=(5 * N, 4.5))
        if N == 1:
            axes = [axes]
            
        for ax, step in zip(axes, steps):
            img_path = self.tmp_path / f'distribution_step_{step}.png'
            if img_path.exists():
                img = mpimg.imread(str(img_path))
                ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'{step}', fontsize=14)
            
        # Draw a line representing the 'steps' axis above all subplots
        # We can use an arrow to indicate direction of steps
        fig.add_artist(mpl.lines.Line2D([0.1, 0.9], [0.9, 0.9], transform=fig.transFigure, color="black", linewidth=2))
        # Add an arrowhead
        fig.add_artist(mpl.patches.RegularPolygon((0.9, 0.9), 3, radius=0.01, orientation=-np.pi/2, transform=fig.transFigure, color="black"))
        
        fig.text(0.5, 0.92, 'Denoising Steps', ha='center', va='bottom', fontsize=16, fontweight='bold')

        plt.tight_layout()
        fig.subplots_adjust(top=0.8)
        
        out_dir = save_path if save_path else self.tmp_path
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / 'distribution_all_steps_report.png'
        
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
