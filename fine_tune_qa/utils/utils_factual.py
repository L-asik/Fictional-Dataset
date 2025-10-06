import os 
import json
import torch
import wandb
from transformers import TrainerCallback
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling, set_seed
from utils.globs import DEBUG_PRINT_DIR, _IGNORE_INDEX
def save_table_locally(questions, preds, answers, accuracy, filename="validation_results.json"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    results = []
    for q, p, a in zip(questions, preds, answers):
        results.append({
            "question": q,
            "prediction": p,
            "answer": a,
            "accuracy": accuracy 
        })
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved validation results to {filename}")
    except Exception as e:
        print(f"Failed to save validation results: {str(e)}")

def batch_generate(model, tokenizer, inputs, cuda=True, if_base=False, debug_path=None, langs=None, ):
    batch_size = len(inputs)
    tokenizer.pad_token = tokenizer.eos_token
    

    if langs is not None:
        inputs = [f"Question: {x.strip()} Answer in {langs[i]}:" for i, x in enumerate(inputs)]
    else:
        inputs = [f"Question: {x.strip()} Answer:" for x in inputs]

    question = tokenizer(
        inputs,
        padding="longest",
        return_tensors="pt"
    )

    if cuda:
        model = model.cuda()
        question = {k: v.cuda() for k, v in question.items()}
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=question["input_ids"],
            attention_mask=question["attention_mask"],
            max_new_tokens=30,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = outputs[:, question["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    file_path = os.path.join(DEBUG_PRINT_DIR, debug_path, "validation.json")
    if file_path is not None and not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding='utf-8') as f:
            data = {
                "formated_question": inputs,
                "inputs decoded": tokenizer.batch_decode(question["input_ids"], skip_special_tokens=False),
                "question input_ids": question["input_ids"].tolist(),
                "decoded answer": decoded,

            }
            json.dump(data, f, indent=4, ensure_ascii=False)
    return decoded

def compute_metrics(model, tokenizer, dataset, batch_size=1, project_name = "validation_results.json", results_path="", if_base=False, current_epoch=0, debug_path=None):
    model.eval()
    accuracy = {}
    for val_set in dataset:
        questions = dataset[val_set][0]
        answers = dataset[val_set][1]
        tags = dataset[val_set][2]
        if len(dataset[val_set]) == 4:
            langs = dataset[val_set][3]
        else:
            langs = None
        preds = []
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            pred_batch = batch_generate(model, tokenizer, batch_questions, if_base=if_base, debug_path=debug_path, langs=langs)
            preds.extend(pred_batch)
            # print(f"Processed {min(i+batch_size, len(questions))}/{len(questions)} samples")
        
        sum_correct = 0
        for i in range(len(preds)):
            if tags[i][-1] in preds[i]:
                print("Tag found in prediction:", tags[i][-1], "in", preds[i])
                sum_correct += 1
        accuracy[val_set] = sum_correct / len(answers)
        
        # Print detailed accuracy report
        print(f"VALIDATION ACCURACY {val_set}: {accuracy [val_set]:.2%}")


    
        # Save locally
        local_filename = f"validation_epoch_{current_epoch}.json"
        local_filename = os.path.join(results_path, project_name, local_filename)
        save_table_locally(questions, preds, answers, accuracy, filename=local_filename)
        
        # Create W&B table
        try:
            table = wandb.Table(columns=["Question", "Prediction", "Answer"])
            for q, p, a in zip(questions, preds, answers):
                table.add_data(q, p, a)
            
            wandb.log({
                f"validation_predictions_vs_answers_epoch{current_epoch}_{val_set}": table,
            })
            print("Successfully logged validation results to W&B")
        except Exception as e:
            print(f"Failed to log to W&B: {str(e)}")
    
    return accuracy
    
class ValidationCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, eval_every=1, accuracy_threshold=0.8, project_name="validation_results.json", if_base=False, results_path="", debug_path = None):
        self.results_path = results_path
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.eval_every = eval_every
        self.threshold = accuracy_threshold
        self.project_name = project_name
        self.if_base = if_base
        self.debug_path = debug_path
        #hardcoded for simplicity, can be parameterized if needed
        self.batch_size = 1 
        print(f"Initialized ValidationCallback with eval_every={eval_every}, threshold={self.threshold}")

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)
        if current_epoch % self.eval_every == 0:
            model = kwargs['model']
            metrics = compute_metrics(
                model, self.tokenizer, self.eval_dataset,
                self.batch_size, self.project_name, 
                results_path=self.results_path,
                if_base=self.if_base,
                current_epoch=current_epoch,
                debug_path=self.debug_path
            )
            
            # Prepare and log accuracy metrics (with epoch/step context)
            log_data = {
                "epoch": current_epoch,
                "global_step": state.global_step,
                **{f"accuracy/{val_set}": acc for val_set, acc in metrics.items()}
            }
            wandb.log(log_data, step=state.global_step)
            

            all_pass_threshold = all(acc >= self.threshold for acc in metrics.values())
            

            if not hasattr(self, "best_accuracies"):
                self.best_accuracies = {k: 0.0 for k in metrics.keys()}
            
            all_improved = True
            for val_set, acc in metrics.items():
                if acc > self.best_accuracies[val_set]:
                    self.best_accuracies[val_set] = acc
                else:
                    all_improved = False
            

            print(f"\nVALIDATION RESULTS - Epoch {current_epoch}")
            for val_set, acc in metrics.items():
                best = self.best_accuracies[val_set]
                print(f"{val_set}: {acc:.2%}")
            
            # Model saving and early stopping logic...
            if all_improved:
                save_path = os.path.join(args.output_dir, "best_model")
                model.save_pretrained(save_path)
            
            if all_pass_threshold:
                control.should_training_stop = True

class CustomDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):

        label_lengths = [len(f["labels"]) for f in features]
        max_length = max(label_lengths)
        

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        
        for f in features:
            pad_len = max_length - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [_IGNORE_INDEX] * pad_len)
        

        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch