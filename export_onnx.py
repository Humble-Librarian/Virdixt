"""
Exports the fine-tuned FinBERT to ONNX format.
"""
import os
import shutil
import argparse
import torch
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
    
    dummy_input = tokenizer("This is a test sentence for ONNX export.", return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]
    
    console.print("Exporting to ONNX...")
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
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
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
        
    tokenizer_src = os.path.join(model_path, 'tokenizer.json')
    if os.path.exists(tokenizer_src):
        shutil.copy(tokenizer_src, os.path.join(output_dir, 'tokenizer.json'))
        
    console.print(f"Export summary: ONNX model saved to {onnx_path}")

if __name__ == '__main__':
    main()
