"""
Exports the fine-tuned FinBERT to ONNX format.
"""
import os
import sys
import shutil
import argparse
import torch

# Ensure UTF-8 stdout/stderr on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import onnx
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Export FinBERT to ONNX format")
    parser.add_argument('--quantize', action='store_true', help='Quantize to INT8')
    args = parser.parse_args()
    
    model_path = './output'
    output_dir = 'models'
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, 'finbert.onnx')
    
    console.print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    dummy_input = tokenizer(
        "This is a test sentence for ONNX export.",
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True
    )
    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]
    
    console.print("Exporting to ONNX...")
    try:
        # Try with dynamo=False (classic reliable TorchScript exporter)
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size'}
            },
            dynamo=False
        )
    except TypeError:
        # Fallback if dynamo param is not accepted
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size'}
            }
        )
    
    console.print("Validating ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    console.print("ONNX model is valid.")
    
    if args.quantize:
        console.print("Quantizing to INT8...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantized_onnx_path = os.path.join(output_dir, 'finbert_quantized.onnx')
        quantize_dynamic(onnx_path, quantized_onnx_path, weight_type=QuantType.QInt8)
        console.print(f"Quantized model saved to {quantized_onnx_path}")
        
    for fname in ['tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'config.json']:
        src = os.path.join(model_path, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, fname))
        
    console.print(f"Export summary: ONNX model and tokenizer assets saved to {output_dir}/")

if __name__ == '__main__':
    main()
