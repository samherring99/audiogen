import argparse

from audiogen.eval.metrics import plot_loss_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", help="Path to checkpoint directory containing trainer_state.json")
    parser.add_argument("--output", default=None, help="Output PNG path (default: <checkpoint_dir>/loss_curve.png)")
    args = parser.parse_args()

    plot_loss_curve(args.checkpoint_dir, args.output)


if __name__ == "__main__":
    main()
