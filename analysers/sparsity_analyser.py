from pathlib import Path
from typing import Optional

import pandas as pd

import torch
import torch.nn as nn
from torch.nn import Module

from analysers.base import AbstractAnalyser


class SparsityAnalyser(AbstractAnalyser):
    """
    Analyser for tracking the sparsity (percentage of exact zeros) across different types of layers
    such as linear layers, activation functions, and QKV projections.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analytical_points = {}

    def add_analytical_point(self, model_name: str, model: Module, *args, **kwargs):
        """
        Calculates and stores the sparsity for different modules in the given model step.
        """
        layer_sparsities = []
        activation_sparsities = []
        qkv_sparsities = []

        for name, module in model.named_modules():
            in_act = getattr(module, 'in_activation', None)
            out_act = getattr(module, 'out_activation', None)

            if in_act is None and out_act is None:
                continue
            
            is_activation = issubclass(module, (nn.SiLU, nn.GELU, nn.ReLU))
            is_qkv = any(qkv_name in name.lower() for qkv_name in ['q_proj', 'k_proj', 'v_proj', 'qkv', 'query', 'key', 'value'])
            
            with torch.no_grad():
                # For activations, use post-layer (out_activation)
                if is_activation and out_act is not None:
                    sparsity = (out_act == 0).float().mean().item() * 100
                    activation_sparsities.append(sparsity)
                else:
                    # For layers and QKV, use pre-layer (in_activation)
                    if in_act is not None:
                        # in_activation might be a tuple if multiple inputs, but decorators.py stores single `x`
                        if isinstance(in_act, tuple):
                            in_act = in_act[0]
                        sparsity = (in_act == 0).float().mean().item() * 100
                        
                        if is_qkv:
                            qkv_sparsities.append(sparsity)
                        else:
                            layer_sparsities.append(sparsity)

        self.analytical_points[model_name] = {
            'Layer Pre-Sparsity (%)': sum(layer_sparsities) / len(layer_sparsities) if layer_sparsities else 0.0,
            'Activation Post-Sparsity (%)': sum(activation_sparsities) / len(activation_sparsities) if activation_sparsities else 0.0,
            'QKV Proj Pre-Sparsity (%)': sum(qkv_sparsities) / len(qkv_sparsities) if qkv_sparsities else 0.0,
        }

    def make_report(self, save_path: Optional[Path] = None, *args, **kwargs):
        """
        Generates a table (CSV/Markdown) of the overall sparsity.
        """
        if not self.analytical_points:
            return

        df = pd.DataFrame.from_dict(self.analytical_points, orient='index')
        df.index.name = 'Model Name'
        
        print("\n=== Sparsity Report ===")
        print(df.to_markdown())

        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path / 'sparsity_report.csv')
            with open(save_path / 'sparsity_report.md', 'w') as f:
                f.write(df.to_markdown())
