import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Train model with LoRA")
    parser.add_argument("--name", type=str, default="",)
    parser.add_argument("--project_name", type=str, default="llama-qa-validation_clika_mod",
                    help="Weights & Biases project name")
    # Model/training arguments
    parser.add_argument("--val_data_trans", type=str, default="")
    parser.add_argument("--test_data_trans", type=str, default="")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-13b-hf",
                       help="Pretrained model name or path")
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training data JSON file")
    parser.add_argument("--val_data", type=str, required=True,
                       help="Path to validation data JSON file")
    parser.add_argument("--test_data", type=str, required=True,
                       help="Path to test data JSON file")
    parser.add_argument("--output_dir", type=str, default="./llama-qa-clika-mod",
                       help="Output directory for saved models")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1000,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=128,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--eval_every", type=int, default=5,
                       help="Evaluate every N epochs")
    parser.add_argument("--accuracy_threshold", type=float, default=0.9,
                       help="Accuracy threshold for early stopping")
    parser.add_argument("--lora_modules", type=str, default="q_proj,v_proj",
                    help="Comma-separated list of modules to apply LoRA to (e.g., 'q_proj,v_proj,k_proj,o_proj')")
    parser.add_argument("--results_path", type=str, default="",
                    help="Path to save validation results")
    parser.add_argument("--afp", action="store_true",
                    help="If the afp pipeline was used")
    parser.add_argument("--checkpoint", type=str, default="",
                    help="path to the afp model")
    parser.add_argument("--add_language_token", action="store_true"  )
    parser.add_argument("--cache_dir", type=str, default="",
                    help="Path to cache directory for model/tokenizer")
    return parser.parse_args()