from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss_curve(checkpoint_dir: str | Path, output_path: str | Path | None = None) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        # Try finding the latest checkpoint subdirectory
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
        if checkpoints:
            state_path = checkpoints[-1] / "trainer_state.json"

    if not state_path.exists():
        raise FileNotFoundError(f"No trainer_state.json found in {checkpoint_dir}")

    with open(state_path) as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in log_history:
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    fig, ax = plt.subplots(figsize=(10, 5))
    if train_losses:
        ax.plot(train_steps, train_losses, label="Train Loss", alpha=0.8)
    if eval_losses:
        ax.plot(eval_steps, eval_losses, label="Eval Loss", marker="o", markersize=4)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if eval_losses:
        best_eval = min(eval_losses)
        best_step = eval_steps[eval_losses.index(best_eval)]
        final_eval = eval_losses[-1]
        overfitting = final_eval > 1.1 * best_eval
        print(f"Best eval loss: {best_eval:.4f} at step {best_step}")
        print(f"Final eval loss: {final_eval:.4f}")
        if overfitting:
            print("Warning: possible overfitting detected (final eval loss > 1.1x best)")

    out = Path(output_path) if output_path else checkpoint_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"Saved to {out}")
