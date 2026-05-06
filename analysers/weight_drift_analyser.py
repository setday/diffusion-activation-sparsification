from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from torch.nn import Module

from analysers.base import AbstractAnalyser


class WeightDriftAnalyser(AbstractAnalyser):
    """
    Analyser for comparing the weights of two models to identify any drift that may have occurred during training or fine-tuning. This analyser can be used to assess the stability of a model's weights and to identify any significant changes that may indicate overfitting or other issues.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.analytical_points = {}  # Dictionary to store analytical points, e.g., weight differences at each step

    def add_analytical_point(self, step: int, model_original: Module, model_moddified: Module, model_moddified_name: str, *args, **kwargs):
        """
        Add an analytical point to the analyser. This method should be implemented by subclasses to define how analytical points are added and what information they contain.
        """
        self.analytical_points[(model_moddified_name, step)] = {}  # Initialize a dictionary for the current step and model

        weight_diff_sum = 0.0
        z_factor_sum = 0.0
        count = 0

        allowed_parameters = ['weight', 'bias']  # Define allowed parameter types to compare
        # iterate trough models with weights and compare them, add the difference to the analytical points
        for (name_a, param_a), (name_b, param_b) in zip(model_original.named_parameters(), model_moddified.named_parameters()):
            assert name_a == name_b, f"Model parameters do not match: {name_a} vs {name_b}"

            for param_type in allowed_parameters:
                if param_type in name_a:
                    break
            else:
                continue  # Skip parameters that are not in the allowed list

            weight_diff = (param_a - param_b).abs().mean()  # Calculate mean absolute difference
            z_factor = 1 - 3 * (param_a.std() + param_b.std()) / (param_a.mean() - param_b.mean()).abs()  # Calculate z-factor for normalization

            weight_diff_sum += weight_diff.item()
            z_factor_sum += z_factor.item()
            count += 1
        
        self.analytical_points[(model_moddified_name, step)] = (
            weight_diff_sum / count if count > 0 else 0,
            z_factor_sum / count if count > 0 else 0
        )

    def make_report(self, save_path: Optional[Path] = None, *args, **kwargs):
        """
        Generate a report based on the analytical points collected by the analyser. This method should be implemented by subclasses to define how the report is generated and what information it contains.
        """
        if save_path is not None:
            # Weight differences plot
            plt.figure(figsize=(10, 5))
            for model_name in set(name for name, _ in self.analytical_points.keys()):
                steps = [0] + [step for (name, step) in self.analytical_points.keys() if name == model_name]
                weight_diffs = [0] + [self.analytical_points[(name, step)][0] for (name, step) in self.analytical_points.keys() if name == model_name]
                plt.plot(steps, weight_diffs, label=f'{model_name} Weight Diff')
            plt.xlabel('Step')
            plt.ylabel('Mean Absolute Weight Difference')
            plt.title('Weight Drift Over Time')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(save_path / 'weight_drift.png')
            plt.close()

            # Z-factors plot
            plt.figure(figsize=(10, 5))
            for model_name in set(name for name, _ in self.analytical_points.keys()):
                steps = [0] + [step for (name, step) in self.analytical_points.keys() if name == model_name]
                z_factors = [1.0] + [self.analytical_points[(name, step)][1] for (name, step) in self.analytical_points.keys() if name == model_name]
                plt.plot(steps, z_factors, label=f'{model_name} Z-Factor')
            plt.xlabel('Step')
            plt.ylabel('Z-Factor')
            plt.title('Z-Factor Over Time')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(save_path / 'z_factors.png')
            plt.close()


        return self.analytical_points
