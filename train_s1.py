from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, concatenate_datasets
from datetime import datetime
timestamp_str = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
import argparse
from graph_vocab.graph_vocabulary import GraphVocabulary
graph_vocab = GraphVocabulary()
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen3-8B")
parser.add_argument("--dmc_dataset_path", type=str, default="data/stage1/GraphVocab_Stage1_DMC_Relabels-15_MaxNodes-5_Train.jsonl")
# parser.add_argument("--flatten_dataset_path", type=str, default="GraphVocab_Stage1_SFT_Flatten_Relabels-100_Nodes-4-12_RandomGraphs-4000_Train.jsonl")
parser.add_argument("--extend_graph_vocab", type=int, default=1, help="0 = No GraphVocab, 1 = Add GraphVocab")
parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device for training.")
parser.add_argument("--save_steps", type=int, default=200, help="Number of steps to save model.")
args = parser.parse_args()

if __name__ == "__main__":
    print(args)
    # Load JSONL file
    dataset_decomp_merge = load_dataset("json", data_files = args.dmc_dataset_path)
    # dataset_decomp_merge = load_dataset("json", data_files="./data/" + args.dmc_dataset_path)
    train_test_split = dataset_decomp_merge["train"].train_test_split(
        test_size=1000,
        shuffle=True,
        seed=42
    )
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    print("Sample from Decomp-Merge Dataset:", dataset_decomp_merge['train'][0])
    print("Sample from Eval Dataset:", eval_dataset[0])
    print("Sample from Train Dataset:", train_dataset[0])

    model_path = args.base_model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True, use_fast=True)
    print(f"Original vocab size: {len(tokenizer)}")


    if args.extend_graph_vocab == 1:
        tokenizer.add_tokens(graph_vocab.GRAPH_TOKENS)
    print(f"New vocab size: {len(tokenizer)}")

    for token in graph_vocab.GRAPH_TOKENS:
        encoded = tokenizer.tokenize(token)
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"{token} -> {encoded} -> {token_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    )
    if args.extend_graph_vocab == 1:
        # Resize token embeddings to accommodate new graph tokens
        model.resize_token_embeddings(len(tokenizer))

    save_path = f"model/{args.base_model_path}-stage1-sft-{timestamp_str}"
    sft_config = SFTConfig(
        output_dir=save_path,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_safetensors=True,
        save_only_model=True,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=args.save_steps,
        save_strategy="steps",
        bf16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=0.5,
        dataset_text_field="messages", 
        packing=False,
        neftune_noise_alpha=5.0,
        eval_strategy="steps",
        dataloader_num_workers=8,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_steps=10,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=sft_config,
    )

    # Start training
    trainer.train()

    # Save final model and tokenizer
    print(timestamp_str)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print("Model and tokenizer saved at: " + save_path)

    