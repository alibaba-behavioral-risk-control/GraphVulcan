from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, concatenate_datasets
from datetime import datetime
import argparse
from graph_vocab.graph_vocabulary import GraphVocabulary
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# timestamp for output paths
timestamp_str = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

graph_vocab = GraphVocabulary()

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B", help="Base model path")
parser.add_argument("--task", type=str, default="s2_joint", help="Task name for output folder")
parser.add_argument("--encoding", type=str, default="GraphVocab", choices=["GraphVocab", "EdgeList", "Incident"], help="Graph encoding format for dataset selection")
parser.add_argument("--cot", action="store_true", default=False, help="Use Chain-of-Thought (CoT) datasets instead of Direct")
parser.add_argument("--per_device_train_batch_size", type=int, default=5, help="Batch size per device for training")
parser.add_argument("--save_steps", type=int, default=200, help="Number of steps to save model")
parser.add_argument("--shuffle_seed", type=int, default=None, help="Seed for shuffling the merged dataset (set None to disable)")
parser.add_argument("--report_to", type=str, default="none", choices=["tensorboard", "none"], help="Report to tensorboard or not")
parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config path")
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
args = parser.parse_args()


def load_joint_dataset(encoding: str = "GraphVocab", cot: bool = False, shuffle_seed: int = None):
    cot_suffix = "_CoT" if cot else "_None"
    # (directory, task_name, node_range, samples, splits)
    task_configs = [
        ("s2_connectivity", "Connectivity", "Nodes-11-50", "Samples-10000", "Splits-2"),
        ("s2_degree", "Degree", "Nodes-11-50", "Samples-10000", "Splits-2"),
        ("s2_cycle_detection", "CycleDetection", "Nodes-11-50", "Samples-10000", "Splits-2"),
        ("s2_isomorphism", "Isomorphism", "Nodes-6-12", "Samples-10000", "Splits-2"),
        ("s2_shortest_path", "ShortestPath", "Nodes-11-50", "Samples-10000", "Splits-2"),
        ("s2_max_common_subgraph", "MCS", "Nodes-5-10", "Samples-10000", "Splits-2"),
        ("s2_max_clique", "MaxClique", "Nodes-5-10", "Samples-10000", "Splits-2"),
    ]
    paths = []
    for directory, task_name, node_range, samples, splits in task_configs:
        filename = f"{encoding}_Stage2_{task_name}{cot_suffix}_{node_range}_{samples}_{splits}_Train.jsonl"
        paths.append(f"{directory}/{filename}")
    print(f"Loading joint dataset with encoding={encoding}, cot={cot}")
    datasets = []
    for p in paths:
        full_path = "./data/" + p
        # full_path = "data/" + p  # for local testing
        ds = load_dataset("json", data_files=full_path, split="train")
        print(f"Loaded dataset {full_path} with {len(ds)} samples")
        datasets.append(ds)
    if not datasets:
        raise ValueError("No datasets loaded. Check dataset_paths.")
    if len(datasets) == 1:
        combined = datasets[0]
    else:
        combined = concatenate_datasets(datasets)
    if shuffle_seed is not None:
        print(f"Shuffling dataset with seed {shuffle_seed}")
        combined = combined.shuffle(seed=shuffle_seed)
    return combined


if __name__ == "__main__":
    print(args)
    base_model_path = f"model/{args.model_path}"
    dataset = load_joint_dataset(encoding=args.encoding, cot=args.cot, shuffle_seed=args.shuffle_seed)
    print("Sample from Joint Dataset:", dataset[0])

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    print(f"Vocab size: {len(tokenizer)}")

    # Check if graph tokens are already in the tokenizer vocabulary
    missing_tokens = [t for t in graph_vocab.GRAPH_TOKENS if t not in tokenizer.get_vocab()]
    if missing_tokens:
        print(f"Detected {len(missing_tokens)} missing graph tokens, extending vocabulary...")
        tokenizer.add_tokens(graph_vocab.GRAPH_TOKENS)
        print(f"Extended vocab size: {len(tokenizer)}")

    for token in graph_vocab.GRAPH_TOKENS:
        encoded = tokenizer.tokenize(token)
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"{token} -> {encoded} -> {token_id}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
    )

    if missing_tokens:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized model embeddings to {len(tokenizer)}")

    model_save_path = f"model/stage2"
    save_path = model_save_path + f"{args.model_path}-stage2-sft-{timestamp_str}"

    if args.deepspeed is not None:
        print(f"Using DeepSpeed config: {args.deepspeed}")
        optim = "adamw_torch"
    else:
        print("DeepSpeed not specified, using 32-bit AdamW")
        optim = "paged_adamw_32bit"

    if args.report_to == "tensorboard":
        print("Setting up TensorBoard logging")
        logging_dir = os.path.join(save_path, "logs")
        print(f"Logging directory: {logging_dir}")
    else:
        logging_dir = None

    sft_config = SFTConfig(
        output_dir=save_path,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_safetensors=True,
        save_only_model=True,
        save_on_each_node=False,  # Only save on main node
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=1,
        logging_steps=10,
        max_length=None,
        neftune_noise_alpha=5.0,
        save_steps=args.save_steps,
        save_strategy="steps",
        bf16=True,
        optim=optim,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to=args.report_to,
        gradient_checkpointing=True,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataset_text_field="messages",
        packing=False,
        dataloader_prefetch_factor=2,
        logging_dir=logging_dir,
        # eval_strategy="steps",
        # eval_steps=100,
        # per_device_eval_batch_size=6,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        logging_first_step=True,
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
    )

    trainer = SFTTrainer(
        model=model,
        # train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=sft_config,
    )

    trainer.train()

    print(timestamp_str)
    # Save final model and tokenizer (only on main process)
    # In distributed training, only rank 0 should save the model
    if trainer.is_world_process_zero():
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        print("Model and tokenizer saved at: " + save_path)
    else:
        print(f"Skipping model save on rank {trainer.args.local_rank}")
