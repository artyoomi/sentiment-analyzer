"""
CLI interface for sentiment analysis.
"""

import argparse
import os
import sys

from analyzer import (
    train_model,
    predict_sentiment,
    load_sentiment_classifier,
    SentimentClassifier,
)
from transformers import AutoTokenizer


def add_parse_args(parser: argparse.ArgumentParser):
    """
    Function to construct CLI interface structure.
    """

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser(
        "train",
        help="Train classifier on geo-reviews-2023 dataset"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Amount of epochs for training (default: 3)"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    train_parser.add_argument(
        "--output",
        default="./results",
        help="Directory to save train results (default: ./results)"
    )
    train_parser.add_argument(
        "--save-dir",
        default="./pretrained",
        help="Directory to save pretrained model (default: ./pretrained)"
    )
    train_parser.add_argument(
        "--freeze-bert",
        action="store_true",
        help="Freeze weights of BERT (train only classifier)"
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict text sentiment"
    )
    predict_parser.add_argument(
        "--text",
        required=True,
        help="Text to analyze"
    )
    predict_parser.add_argument(
        "--model-path",
        default="./pretrained",
        help="Path to saved model directory (default: ./pretrained)"
    )


def _has_inference_checkpoint(model_path: str) -> bool:
    return os.path.isfile(os.path.join(model_path, "pytorch_model.bin")) or os.path.isfile(
        os.path.join(model_path, "model.safetensors")
    )


def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train --epochs 5
  %(prog)s predict --text "Good service!"
        """
    )
    add_parse_args(parser)
    args = parser.parse_args()

    if args.command == "train":
        print(f"Start of training...")
        print(f"  Epochs: {args.epochs}")
        print(f"  Freeze BERT: {args.freeze_bert}")

        model, tokenizer = train_model(
            output_dir=args.output,
            save_dir=args.save_dir,
            num_epochs=args.epochs,
            freeze_bert=args.freeze_bert,
        )
        print("\nTraining completed!")

    elif args.command == "predict":
        if not os.path.isdir(args.model_path) or not _has_inference_checkpoint(args.model_path):
            print(
                f"Error: no trained weights in {args.model_path!r} "
                "(need pytorch_model.bin or model.safetensors).",
                file=sys.stderr,
            )
            print("Train first: python cli.py train", file=sys.stderr)
            sys.exit(1)

        print(f"Loading model from {args.model_path}...")

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = load_sentiment_classifier(args.model_path)

        rating = predict_sentiment(args.text, model, tokenizer)

        # Result output (6 classes from 0 to 5)
        sentiment_map = {
            0: "Very negative",
            1: "Negative",
            2: "Mostly negative",
            3: "Mostly positive",
            4: "Positive",
            5: "Very positive"
        }
        sentiment = sentiment_map.get(rating, "Unknown")

        print(f"\nAnalysis result:")
        print(f"  Text: {args.text[:100]}{'...' if len(args.text) > 100 else ''}")
        print(f"  Predicted rating: {rating[0]}/5, confidence: {(rating[1] * 100):.2f}%")
        print(f"  Sentiment: {sentiment}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
