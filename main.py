# main.py

import argparse
from data_collection import collect_demonstrations
from training import train_model
from evaluation import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Imitation Learning for Overcooked AI")
    parser.add_argument('--collect', action='store_true', help='Collect expert demonstrations')
    parser.add_argument('--train', action='store_true', help='Train the imitation learning model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the trained model')
    args = parser.parse_args()
    
    if args.collect:
        collect_demonstrations()
    if args.train:
        train_model()
    if args.evaluate:
        evaluate_model()

if __name__ == "__main__":
    main()
