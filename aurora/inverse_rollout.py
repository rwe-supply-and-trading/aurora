"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Inverse rollout functionality for computing initial condition sensitivities.

This module provides tools for backpropagating perturbations from model outputs
back to initial conditions, enabling sensitivity analysis and inverse problems.
"""

import dataclasses
from typing import Callable, Optional

import torch
from torch import Tensor

from aurora.batch import Batch
from aurora.model.aurora import Aurora

__all__ = [
    "differentiable_rollout",
    "enable_batch_gradients",
    "compute_initial_perturbation",
    "extract_timeseries",
    "create_trajectory_perturbation_loss",
    "InverseRolloutSolver",
]


def enable_batch_gradients(batch: Batch, include_static: bool = False) -> Batch:
    """Enable gradient tracking on all tensor fields in a Batch.

    Args:
        batch: The batch to enable gradients on.
        include_static: Whether to enable gradients on static variables. Defaults to False
            since static variables (topography, land-sea mask, etc.) typically don't need
            gradients.

    Returns:
        A new Batch with requires_grad=True on all variable tensors.
    """

    def _enable_grad(t: Tensor) -> Tensor:
        return t.detach().clone().requires_grad_(True)

    new_batch = dataclasses.replace(
        batch,
        surf_vars={k: _enable_grad(v) for k, v in batch.surf_vars.items()},
        atmos_vars={k: _enable_grad(v) for k, v in batch.atmos_vars.items()},
    )

    if include_static:
        new_batch = dataclasses.replace(
            new_batch,
            static_vars={k: _enable_grad(v) for k, v in batch.static_vars.items()},
        )

    return new_batch


def differentiable_rollout(
    model: Aurora,
    batch: Batch,
    steps: int,
    extract_fn: Optional[Callable[[Batch, int], Tensor]] = None,
) -> tuple[list[Batch], Optional[list[Tensor]]]:
    """Perform a rollout while retaining the computation graph for backpropagation.

    Unlike the generator-based `rollout()`, this stores all predictions in memory
    to enable gradient computation back to the initial state.

    Args:
        model: The Aurora model.
        batch: Initial state (will have gradients enabled on its tensors if they have
            requires_grad=True).
        steps: Number of rollout steps.
        extract_fn: Optional function to extract values of interest from each step.
            Signature: (pred_batch, step_index) -> Tensor. Useful for extracting
            specific variables or grid points during the rollout.

    Returns:
        Tuple of:
        - list of predicted Batches (one per step)
        - list of extracted tensors if extract_fn provided, otherwise None
    """
    # Prepare batch (similar to rollout.py but keep grad tracking)
    batch = model.batch_transform_hook(batch)

    # Get model dtype for mixed precision
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    predictions: list[Batch] = []
    extracted: Optional[list[Tensor]] = [] if extract_fn is not None else None

    for step in range(steps):
        pred = model.forward(batch)

        predictions.append(pred)

        if extract_fn is not None:
            extracted.append(extract_fn(pred, step))

        # Create next input batch (keeping computation graph intact)
        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )

    return predictions, extracted


def extract_timeseries(
    predictions: list[Batch],
    var_name: str,
    lat_idx: int,
    lon_idx: int,
    var_type: str = "surf",
    level_idx: Optional[int] = None,
    batch_idx: int = 0,
) -> Tensor:
    """Extract a time series of a variable at a single grid point from predictions.

    Args:
        predictions: List of predicted Batch objects from rollout.
        var_name: Name of the variable (e.g., "2t" for 2m temperature).
        lat_idx: Latitude index in the grid.
        lon_idx: Longitude index in the grid.
        var_type: "surf" for surface variables, "atmos" for atmospheric.
        level_idx: Pressure level index (required for atmospheric variables).
        batch_idx: Index into the batch dimension. Defaults to 0.

    Returns:
        Tensor of shape (steps,) containing the time series.
    """
    values = []
    for pred in predictions:
        if var_type == "surf":
            # Surface vars shape: (batch, time=1, height, width)
            val = pred.surf_vars[var_name][batch_idx, 0, lat_idx, lon_idx]
        elif var_type == "atmos":
            # Atmos vars shape: (batch, time=1, levels, height, width)
            if level_idx is None:
                raise ValueError("level_idx required for atmospheric variables")
            val = pred.atmos_vars[var_name][batch_idx, 0, level_idx, lat_idx, lon_idx]
        else:
            raise ValueError(f"var_type must be 'surf' or 'atmos', got '{var_type}'")
        values.append(val)

    return torch.stack(values)


def compute_initial_perturbation(
    model: Aurora,
    initial_batch: Batch,
    steps: int,
    loss_fn: Callable[[list[Batch]], Tensor],
    return_predictions: bool = False,
) -> dict:
    """Compute the initial condition perturbation that would produce a desired output change.

    This performs a forward rollout, computes a loss on the trajectory, and backpropagates
    to find the gradient with respect to the initial state. The gradient indicates the
    direction to perturb the initial state to increase the loss.

    Args:
        model: The Aurora model.
        initial_batch: The original initial state.
        steps: Number of rollout steps.
        loss_fn: Function that takes the list of predictions and returns a scalar loss.
            This encodes the "perturbation" - e.g., a function that measures how much
            a specific variable at a specific point deviates from a target.
        return_predictions: Whether to return the forward predictions as well.

    Returns:
        Dictionary containing:
        - 'surf_var_grads': Gradients for surface variables {var_name: Tensor}
        - 'atmos_var_grads': Gradients for atmospheric variables {var_name: Tensor}
        - 'loss': The computed loss value
        - 'predictions': List of predicted Batches (if return_predictions=True)
    """
    # Enable gradients on the initial state
    batch_with_grad = enable_batch_gradients(initial_batch)

    # Forward pass with gradient tracking
    predictions, _ = differentiable_rollout(model, batch_with_grad, steps)

    # Compute loss (scalar representing how much trajectory deviates from target)
    loss = loss_fn(predictions)

    # Backpropagate to initial state
    loss.backward()

    # Extract gradients
    result = {
        "surf_var_grads": {
            k: v.grad.clone() if v.grad is not None else None
            for k, v in batch_with_grad.surf_vars.items()
        },
        "atmos_var_grads": {
            k: v.grad.clone() if v.grad is not None else None
            for k, v in batch_with_grad.atmos_vars.items()
        },
        "loss": loss.item(),
    }

    if return_predictions:
        result["predictions"] = predictions

    return result


def create_trajectory_perturbation_loss(
    var_name: str,
    lat_idx: int,
    lon_idx: int,
    perturbation_direction: Tensor,
    var_type: str = "surf",
    level_idx: Optional[int] = None,
    step_indices: Optional[list[int]] = None,
    batch_idx: int = 0,
) -> Callable[[list[Batch]], Tensor]:
    """Create a loss function for sensitivity analysis in a specified direction.

    The loss computes the inner product of the trajectory with the perturbation direction.
    The gradient of this loss gives the sensitivity of the trajectory to initial conditions
    in that direction - i.e., which initial state changes would most efficiently move
    the trajectory in the perturbation direction.

    Args:
        var_name: Variable name (e.g., "2t" for 2m temperature).
        lat_idx: Latitude index in the grid.
        lon_idx: Longitude index in the grid.
        perturbation_direction: Tensor specifying the direction of desired perturbation.
            For example, torch.tensor([-1.0, -1.0, -1.0]) to find sensitivities for
            decreasing the variable over 3 time steps.
        var_type: "surf" for surface variables, "atmos" for atmospheric.
        level_idx: Level index for atmospheric variables.
        step_indices: Which rollout steps to include in the loss. If None, uses all steps
            matching the length of perturbation_direction.
        batch_idx: Index into the batch dimension. Defaults to 0.

    Returns:
        Loss function suitable for compute_initial_perturbation.
    """

    def loss_fn(predictions: list[Batch]) -> Tensor:
        # Extract the time series at the point of interest
        timeseries = extract_timeseries(
            predictions, var_name, lat_idx, lon_idx, var_type, level_idx, batch_idx
        )

        # Determine which steps to use
        if step_indices is not None:
            indices = step_indices
        else:
            indices = list(range(len(perturbation_direction)))

        selected = timeseries[indices]
        direction = perturbation_direction[: len(indices)].to(
            selected.device, selected.dtype
        )

        # Inner product with perturbation direction.
        # The gradient of this gives the sensitivity: which initial state changes
        # would most efficiently move the trajectory in this direction.
        loss = torch.sum(selected * direction)

        return loss

    return loss_fn


class InverseRolloutSolver:
    """Solver for finding initial perturbations that produce desired output changes.

    This uses gradient descent to iteratively refine the initial perturbation,
    minimizing the difference between the achieved trajectory change and the
    target trajectory change.

    Example:
        >>> solver = InverseRolloutSolver(model, initial_batch, steps=10)
        >>> # Find perturbation that makes temperature 5K lower at steps 3-7
        >>> target_delta = torch.tensor([-5.0, -5.0, -5.0, -5.0, -5.0])
        >>> solution = solver.solve(
        ...     target_trajectory_delta=target_delta,
        ...     var_name="2t",
        ...     lat_idx=100,
        ...     lon_idx=200,
        ... )
        >>> # solution['surf_var_perturbations']['2t'] contains the optimized perturbation
    """

    def __init__(
        self,
        model: Aurora,
        reference_batch: Batch,
        steps: int,
    ):
        """Initialize the solver.

        Args:
            model: The Aurora model.
            reference_batch: The reference initial state (unperturbed).
            steps: Number of rollout steps.
        """
        self.model = model
        self.reference_batch = reference_batch
        self.steps = steps
        self._reference_predictions: Optional[list[Batch]] = None

    def _ensure_reference_predictions(self) -> list[Batch]:
        """Compute and cache the reference trajectory."""
        if self._reference_predictions is None:
            self.model.eval()
            with torch.no_grad():
                self._reference_predictions, _ = differentiable_rollout(
                    self.model, self.reference_batch, self.steps
                )
        return self._reference_predictions

    @property
    def reference_predictions(self) -> list[Batch]:
        """The reference trajectory (computed lazily)."""
        return self._ensure_reference_predictions()

    def solve(
        self,
        target_trajectory_delta: Tensor,
        var_name: str,
        lat_idx: int,
        lon_idx: int,
        var_type: str = "surf",
        level_idx: Optional[int] = None,
        step_indices: Optional[list[int]] = None,
        batch_idx: int = 0,
        learning_rate: float = 1e-3,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        regularization: float = 1e-4,
        variables_to_perturb: Optional[dict[str, list[str]]] = None,
        verbose: bool = False,
    ) -> dict:
        """Solve for initial perturbations using gradient-based optimization.

        Args:
            target_trajectory_delta: The desired change in the output trajectory at the
                specified point. Shape should be (num_steps,) where num_steps is either
                len(step_indices) or self.steps.
            var_name: Variable to track in the output trajectory.
            lat_idx: Latitude index of the grid point to track.
            lon_idx: Longitude index of the grid point to track.
            var_type: "surf" for surface variables, "atmos" for atmospheric.
            level_idx: Level index for atmospheric variables.
            step_indices: Which rollout steps to apply the target delta to. If None,
                uses all steps matching target_trajectory_delta length.
            batch_idx: Index into the batch dimension. Defaults to 0.
            learning_rate: Optimization learning rate.
            max_iterations: Maximum optimization iterations.
            tolerance: Convergence tolerance on loss change.
            regularization: L2 regularization weight on initial perturbation magnitude.
                Encourages finding the smallest perturbation that achieves the target.
            variables_to_perturb: Which variables to allow perturbations on. Dict with
                keys 'surf' and/or 'atmos', values are lists of variable names.
                If None, all variables can be perturbed.
            verbose: Print progress during optimization.

        Returns:
            Dictionary with:
            - 'surf_var_perturbations': Optimized perturbations for surface variables
            - 'atmos_var_perturbations': Optimized perturbations for atmospheric variables
            - 'losses': List of loss values during optimization
            - 'trajectory_losses': List of trajectory matching losses
            - 'reg_losses': List of regularization losses
            - 'iterations': Number of iterations run
            - 'converged': Whether the optimization converged
            - 'final_trajectory_delta': The achieved trajectory change
        """
        # Determine which steps to target
        if step_indices is None:
            step_indices = list(range(len(target_trajectory_delta)))

        # Ensure reference predictions exist
        self._ensure_reference_predictions()

        # Extract reference trajectory at the target point
        reference_trajectory = extract_timeseries(
            self.reference_predictions,
            var_name,
            lat_idx,
            lon_idx,
            var_type,
            level_idx,
            batch_idx,
        )

        # Determine which variables to perturb
        if variables_to_perturb is None:
            surf_vars_to_perturb = list(self.reference_batch.surf_vars.keys())
            atmos_vars_to_perturb = list(self.reference_batch.atmos_vars.keys())
        else:
            surf_vars_to_perturb = variables_to_perturb.get("surf", [])
            atmos_vars_to_perturb = variables_to_perturb.get("atmos", [])

        # Initialize perturbation tensors (to be optimized)
        p = next(self.model.parameters())
        perturbation = {
            "surf_vars": {
                k: torch.zeros_like(v, device=p.device, dtype=p.dtype, requires_grad=True)
                for k, v in self.reference_batch.surf_vars.items()
                if k in surf_vars_to_perturb
            },
            "atmos_vars": {
                k: torch.zeros_like(v, device=p.device, dtype=p.dtype, requires_grad=True)
                for k, v in self.reference_batch.atmos_vars.items()
                if k in atmos_vars_to_perturb
            },
        }

        # Flatten for optimizer
        params = list(perturbation["surf_vars"].values()) + list(
            perturbation["atmos_vars"].values()
        )

        if not params:
            raise ValueError(
                "No variables selected for perturbation. Check variables_to_perturb."
            )

        optimizer = torch.optim.Adam(params, lr=learning_rate)

        target_delta = target_trajectory_delta.to(p.device, p.dtype)

        losses: list[float] = []
        trajectory_losses: list[float] = []
        reg_losses: list[float] = []

        for iteration in range(max_iterations):
            optimizer.zero_grad()

            # Create perturbed initial state
            perturbed_surf_vars = {
                k: v + perturbation["surf_vars"].get(k, 0)
                for k, v in self.reference_batch.surf_vars.items()
            }
            perturbed_atmos_vars = {
                k: v + perturbation["atmos_vars"].get(k, 0)
                for k, v in self.reference_batch.atmos_vars.items()
            }

            perturbed_batch = dataclasses.replace(
                self.reference_batch,
                surf_vars=perturbed_surf_vars,
                atmos_vars=perturbed_atmos_vars,
            )

            # Forward pass
            predictions, _ = differentiable_rollout(
                self.model, perturbed_batch, self.steps
            )

            # Extract trajectory and compute loss
            current_trajectory = extract_timeseries(
                predictions, var_name, lat_idx, lon_idx, var_type, level_idx, batch_idx
            )

            trajectory_delta = current_trajectory - reference_trajectory.detach()

            # Select the relevant steps
            selected_delta = trajectory_delta[step_indices]
            selected_target = target_delta[: len(step_indices)]

            # MSE loss between achieved delta and target delta
            trajectory_loss = torch.mean((selected_delta - selected_target) ** 2)

            # Regularization to keep perturbations small
            reg_loss = torch.tensor(0.0, device=p.device, dtype=p.dtype)
            for param in params:
                reg_loss = reg_loss + torch.mean(param**2)
            reg_loss = reg_loss * regularization

            loss = trajectory_loss + reg_loss

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            trajectory_losses.append(trajectory_loss.item())
            reg_losses.append(reg_loss.item())

            if verbose and (iteration + 1) % 10 == 0:
                print(
                    f"Iteration {iteration + 1}: loss={loss.item():.6f}, "
                    f"traj_loss={trajectory_loss.item():.6f}, "
                    f"reg_loss={reg_loss.item():.6f}"
                )

            if iteration > 0 and abs(losses[-1] - losses[-2]) < tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

        # Compute final achieved trajectory delta
        with torch.no_grad():
            perturbed_surf_vars = {
                k: v + perturbation["surf_vars"].get(k, 0)
                for k, v in self.reference_batch.surf_vars.items()
            }
            perturbed_atmos_vars = {
                k: v + perturbation["atmos_vars"].get(k, 0)
                for k, v in self.reference_batch.atmos_vars.items()
            }
            perturbed_batch = dataclasses.replace(
                self.reference_batch,
                surf_vars=perturbed_surf_vars,
                atmos_vars=perturbed_atmos_vars,
            )
            final_predictions, _ = differentiable_rollout(
                self.model, perturbed_batch, self.steps
            )
            final_trajectory = extract_timeseries(
                final_predictions,
                var_name,
                lat_idx,
                lon_idx,
                var_type,
                level_idx,
                batch_idx,
            )
            final_trajectory_delta = final_trajectory - reference_trajectory

        return {
            "surf_var_perturbations": {
                k: v.detach().clone() for k, v in perturbation["surf_vars"].items()
            },
            "atmos_var_perturbations": {
                k: v.detach().clone() for k, v in perturbation["atmos_vars"].items()
            },
            "losses": losses,
            "trajectory_losses": trajectory_losses,
            "reg_losses": reg_losses,
            "iterations": iteration + 1,
            "converged": iteration < max_iterations - 1,
            "final_trajectory_delta": final_trajectory_delta.detach(),
            "target_trajectory_delta": target_delta.detach(),
        }


def create_multipoint_loss(
    points: list[dict],
) -> Callable[[list[Batch]], Tensor]:
    """Create a loss function for sensitivity analysis across multiple points.

    This is useful when you want to compute sensitivities for multiple
    variables or grid points simultaneously.

    Args:
        points: List of dictionaries, each specifying a point with keys:
            - var_name: Variable name
            - lat_idx: Latitude index
            - lon_idx: Longitude index
            - perturbation_direction: Tensor of perturbation direction
            - var_type: "surf" or "atmos" (default: "surf")
            - level_idx: Level index for atmospheric variables (optional)
            - step_indices: Which steps to include (optional)
            - batch_idx: Batch index (default: 0)
            - weight: Weight for this point in the total loss (default: 1.0)

    Returns:
        Loss function suitable for compute_initial_perturbation.
    """

    def loss_fn(predictions: list[Batch]) -> Tensor:
        total_loss = None

        for point in points:
            var_name = point["var_name"]
            lat_idx = point["lat_idx"]
            lon_idx = point["lon_idx"]
            perturbation_direction = point["perturbation_direction"]
            var_type = point.get("var_type", "surf")
            level_idx = point.get("level_idx")
            step_indices = point.get("step_indices")
            batch_idx = point.get("batch_idx", 0)
            weight = point.get("weight", 1.0)

            timeseries = extract_timeseries(
                predictions, var_name, lat_idx, lon_idx, var_type, level_idx, batch_idx
            )

            if step_indices is not None:
                indices = step_indices
            else:
                indices = list(range(len(perturbation_direction)))

            selected = timeseries[indices]
            direction = perturbation_direction[: len(indices)].to(
                selected.device, selected.dtype
            )

            point_loss = weight * torch.sum(selected * direction)

            if total_loss is None:
                total_loss = point_loss
            else:
                total_loss = total_loss + point_loss

        return total_loss

    return loss_fn
