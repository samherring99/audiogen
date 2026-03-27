import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def plot_trainer_state(checkpoint_dir: str):
    checkpoint_path = Path(checkpoint_dir)
    state_file = checkpoint_path / "trainer_state.json"

    if not state_file.exists():
        print(f"Error: {state_file} not found")
        return

    with open(state_file, 'r') as f:
        state = json.load(f)

    train_loss = []
    train_steps = []
    eval_loss = []
    eval_steps = []

    for entry in state['log_history']:
        if 'loss' in entry:
            train_steps.append(entry['step'])
            train_loss.append(entry['loss'])
        if 'eval_loss' in entry:
            eval_steps.append(entry['step'])
            eval_loss.append(entry['eval_loss'])

    plt.figure(figsize=(12, 6))

    if train_loss:
        plt.plot(train_steps, train_loss, label='Train Loss', alpha=0.7)
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label='Eval Loss', linewidth=2)

    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    output_file = checkpoint_path / "loss_curve.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")

    if eval_loss:
        print(f"\nCurrent eval loss: {eval_loss[-1]:.4f} at step {eval_steps[-1]}")
    if train_loss:
        print(f"Current train loss: {train_loss[-1]:.4f} at step {train_steps[-1]}")

    if len(eval_loss) > 2:
        if eval_loss[-1] > min(eval_loss):
            min_eval = min(eval_loss)
            min_step = eval_steps[eval_loss.index(min_eval)]
            print(f"\nOverfitting detected!")
            print(f"   Best eval loss: {min_eval:.4f} at step {min_step}")
            print(f"   Current eval loss: {eval_loss[-1]:.4f} (worse by {eval_loss[-1] - min_eval:.4f})")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=str,
                       help="Path to checkpoint directory")
    args = parser.parse_args()

    plot_trainer_state(args.checkpoint_dir)
