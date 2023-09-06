import argparse
from pathlib import Path

def load_train_options():
    parser = argparse.ArgumentParser(description='Command Line options for OCRNet')
    
    parser.add_argument('--train_dataset', type=Path, required=True, help="The dataset path.")
    parser.add_argument('--output_path', type=Path, required=True, help="The saved model output path.")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model', type=Path, help="The model path.")
    
    return parser

def load_test_options():
    parser = argparse.ArgumentParser(description='Command Line options for OCRNet')
    
    parser.add_argument('--test_dataset', type=Path, required=True, help="The dataset path.")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model', type=Path, help="The model path.")
    
    return parser

def load_trt_convert_options():
    parser = argparse.ArgumentParser(description='Command Line options for OCRNet')
    
    parser.add_argument('--input_name', type=Path, required=True, help="The input path.")
    parser.add_argument('--output_name', type=Path, required=True, help="The converted model output path.")
    
    return parser