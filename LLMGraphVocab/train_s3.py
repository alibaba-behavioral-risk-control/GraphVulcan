import argparse
import json
import re
from datetime import datetime
from typing import List, Dict

import torch
import networkx as nx
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import os

from evaluate.evaluate_connectivity import compute_reward_connectivity
from evaluate.evaluate_isomorphism import compute_reward_isomorphism
from evaluate.evaluate_degree import compute_reward_degree
from evaluate.evaluate_shortest_path import compute_reward_shortest_path
from evaluate.evaluate_cycle_detection import compute_reward_cycle_detection
from evaluate.evaluate_max_common_subgraph import compute_reward_mcs
from evaluate.evaluate_max_clique import compute_reward_max_clique
from graph_vocab.graph_tokenizer import GraphTokenizer

TASK_REWARD_FN = {
    "connectivity": compute_reward_connectivity,
    "isomorphism": compute_reward_isomorphism,
    "degree": compute_reward_degree,
    "shortest_path": compute_reward_shortest_path,
    "max_common_subgraph": compute_reward_mcs,
    "cycle_detection": compute_reward_cycle_detection,
    "max_clique": compute_reward_max_clique,
}

# Initialize graph tokenizer for reasoning validation
graph_tokenizer = GraphTokenizer()

def compute_reward_single(task: str, user_msg: str, assistant_msg: str) -> float:
    """Compute reward for a single response using the task-specific reward function."""
    reward_fn = TASK_REWARD_FN.get(task)
    if reward_fn is None:
        print(f"Warning: Unknown task '{task}', returning 0.0")
        return 0.0
    return 2 * reward_fn(user_msg, assistant_msg)

def validate_reasoning_process(user_msg: str, assistant_msg: str) -> float:
    """
    Validate the reasoning process in assistant_msg by checking if all graph tokens
    used in the reasoning are valid (i.e., all edges exist in the original graph).
    
    Args:
        user_msg: The user message containing the original graph
        assistant_msg: The assistant's response containing reasoning with graph tokens
        
    Returns:
        Penalty score (0.0 to N*0.5, where N is the number of invalid graph tokens)
    """
    try:
        # Extract the original graph from user_msg
        original_graph = graph_tokenizer.decode_graph_vocab(user_msg)
        original_edges = set(frozenset(e) for e in original_graph.edges())
        
        # Extract all graph tokens from assistant_msg (excluding those in <think> tags if present)
        # Find all graph token patterns: <NidB>...<NidE><G...>
        graph_token_pattern = r'<NidB>.*?<NidE><G[^>]+>'
        graph_tokens = re.findall(graph_token_pattern, assistant_msg)
        
        if not graph_tokens:
            # No graph tokens found in reasoning, no penalty
            return 0.0
        
        invalid_count = 0
        valid_token_count = 0  # Count tokens with edges (excluding single node tokens)
        
        for graph_str in graph_tokens:
            try:
                # Parse the graph token
                token, nodes = graph_tokenizer.graph_vocab.parse_graph_string(graph_str)
                
                # Handle single node tokens
                if len(nodes) == 1:
                    # Check if this single node is an isolated node in the original graph
                    node = nodes[0]
                    if node in original_graph.nodes():
                        # If the node has degree > 0, it's not isolated, so it's invalid
                        if original_graph.degree(node) > 0:
                            valid_token_count += 1
                            invalid_count += 1
                    continue
                
                # Count this as a valid token to check
                valid_token_count += 1
                
                # Instantiate the subgraph from the token
                subgraph = graph_tokenizer.graph_vocab.instantiate_graph_from_token(token, nodes)
                subgraph_edges = set(frozenset(e) for e in subgraph.edges())
                
                # Check if all edges in this subgraph exist in the original graph
                invalid_edges = subgraph_edges - original_edges
                
                if invalid_edges:
                    # This graph token contains edges not in the original graph
                    invalid_count += 1
                    
            except Exception as e:
                # If we can't parse or validate a token, consider it invalid
                valid_token_count += 1
                invalid_count += 1
                continue
        
        # Calculate penalty based on error ratio
        if valid_token_count == 0:
            # No tokens with edges to validate, no penalty
            return 0.0
        
        # Penalty = ratio of invalid tokens
        penalty = invalid_count / valid_token_count
        return max(0.0, 1.0 - penalty)
        
    except Exception as e:
        # If we can't extract or validate the graph, return no penalty
        # (to avoid penalizing due to parsing errors)
        return 0.0

def compute_reward(completions: List[str], prompts: List[str] = None, tasks: List[str] = None, **kwargs) -> List[float]:
    """
    Compute rewards for a batch of completions.
    
    This function is called by GRPOTrainer with the following signature:
    reward_fn(completions, prompts=prompts, **other_kwargs)
    
    Args:
        completions: List of completion strings (flattened)
        prompts: List of prompt strings (JSON-encoded messages), passed via kwargs
        tasks: List of task names for each prompt, passed via kwargs
        **kwargs: Additional keyword arguments from GRPOTrainer
        
    Returns:
        List of reward scores (1.0 for correct, 0.0 for incorrect)
    """
    if prompts is None:
        # Fallback: try to get prompts from kwargs
        prompts = kwargs.get("prompts", [])
    
    if tasks is None:
        # Fallback: try to get tasks from kwargs
        tasks = kwargs.get("tasks", [])
    
    if not prompts:
        print("Warning: No prompts provided, returning zero rewards")
        return [0.0] * len(completions)
    
    if not tasks or len(tasks) != len(prompts):
        print(f"Warning: Tasks not provided or length mismatch (tasks={len(tasks)}, prompts={len(prompts)}), returning zero rewards")
        return [0.0] * len(completions)
    
    rewards = []
    
    # Calculate how many completions per prompt
    num_completions_per_prompt = len(completions) // len(prompts)
    
    for i, (prompt_str, task) in enumerate(zip(prompts, tasks)):
        # Get the completions for this prompt
        start_idx = i * num_completions_per_prompt
        end_idx = start_idx + num_completions_per_prompt
        prompt_completions = completions[start_idx:end_idx]
        
        # The prompt_str is already the user message (extracted by parse_prompt)
        try:
            # Use prompt_str directly as the user message
            user_msg = prompt_str.strip() if isinstance(prompt_str, str) else str(prompt_str).strip()
            
            if not user_msg:
                # If prompt is empty, assign 0 reward to all completions
                rewards.extend([0.0] * len(prompt_completions))
                continue
            
            # Compute reward for each completion using the task-specific reward function
            for completion in prompt_completions:
                # Get base reward from task-specific reward function (0.0 to 1.0)
                base_reward = compute_reward_single(task, user_msg, completion.strip())
                
                # Validate reasoning process and get reasoning reward (0.0 to 1.0)
                reasoning_reward = validate_reasoning_process(user_msg, completion.strip())
                
                # Final reward = base reward + reasoning reward
                final_reward = base_reward + reasoning_reward
                rewards.append(final_reward)
                
        except Exception as e:
            # If any error occurs, assign 0 reward
            print(f"Error processing prompt with task '{task}': {e}")
            rewards.extend([0.0] * len(prompt_completions))
    
    return rewards

def parse_prompt(example, tokenizer):
    """
    Parse the messages field and convert to prompt format using chat template.
    This ensures consistency between training and inference.
    Handles both 'messages' and 'prompt' fields.
    """
    # If already has 'prompt' field, parse it if it's a string
    if "prompt" in example:
        if isinstance(example["prompt"], str):
            example["prompt"] = json.loads(example["prompt"])
        return example
    
    # If has 'messages' field, convert it to 'prompt'
    if "messages" in example:
        messages = example["messages"]
        # If messages is a string, parse it first
        if isinstance(messages, str):
            messages = json.loads(messages)
        
        # Extract system and user messages to construct input_messages
        input_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                input_messages.append({"role": "system", "content": msg.get("content", "")})
            elif msg.get("role") == "user":
                input_messages.append({"role": "user", "content": msg.get("content", "")})
                break  # Only take the first user message
        
        # Use chat template if available, otherwise fallback to simple concatenation
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            try:
                # Use official chat template (same as inference)
                prompt = tokenizer.apply_chat_template(
                    input_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                example["prompt"] = prompt
            except Exception as e:
                # Fallback to simple concatenation if template fails
                print(f"Warning: Failed to apply chat template: {e}, using fallback")
                system_msg = next((m["content"] for m in input_messages if m["role"] == "system"), "")
                user_msg = next((m["content"] for m in input_messages if m["role"] == "user"), "")
                if system_msg and user_msg:
                    example["prompt"] = f"{system_msg}\n\nUser: {user_msg}\n\nAssistant:"
                elif user_msg:
                    example["prompt"] = f"User: {user_msg}\n\nAssistant:"
                else:
                    example["prompt"] = ""
        else:
            # Fallback: manual concatenation (same as inference fallback)
            system_msg = next((m["content"] for m in input_messages if m["role"] == "system"), "")
            user_msg = next((m["content"] for m in input_messages if m["role"] == "user"), "")
            if system_msg and user_msg:
                example["prompt"] = f"{system_msg}\n\nUser: {user_msg}\n\nAssistant:"
            elif user_msg:
                example["prompt"] = f"User: {user_msg}\n\nAssistant:"
            else:
                example["prompt"] = ""
    
    return example

def load_joint_dataset_interleaved(num_samples_per_task: int = 5000):
    """
    Load all task datasets and interleave them in a round-robin fashion.
    Each task's samples are arranged in order of increasing difficulty.
    The merged dataset cycles through tasks: connectivity, cycle_detection, degree, ...
    
    Args:
        num_samples_per_task: Number of samples per task to load (used to construct dataset paths)
    
    Returns:
        Dataset: Interleaved dataset with all tasks
    """
    # Define dataset path templates for all tasks
    dataset_path_templates = {
        "connectivity": f"data/s2_connectivity/GraphVocab_Stage2_Connectivity_CoT_Nodes-11-50_Samples-{num_samples_per_task}_Splits-1_Train.jsonl",
        "cycle_detection": f"data/s2_cycle_detection/GraphVocab_Stage2_CycleDetection_CoT_Nodes-11-50_Samples-{num_samples_per_task}_Splits-1_Train.jsonl",
        "degree": f"data/s2_degree/GraphVocab_Stage2_Degree_CoT_Nodes-11-50_Samples-{num_samples_per_task}_Splits-1_Train.jsonl",
        "isomorphism": f"data/s2_isomorphism/GraphVocab_Stage2_Isomorphism_CoT_Nodes-6-12_Samples-{num_samples_per_task}_Splits-1_Train.jsonl",
        "shortest_path": f"data/s2_shortest_path/GraphVocab_Stage2_ShortestPath_CoT_Nodes-11-50_Samples-{num_samples_per_task}_Splits-1_Train.jsonl",
        "max_common_subgraph": f"data/s2_max_common_subgraph/GraphVocab_Stage2_MCS_CoT_Nodes-5-10_Samples-{num_samples_per_task}_Splits-1_Train.jsonl",
        "max_clique": f"data/s2_max_clique/GraphVocab_Stage2_MaxClique_CoT_Nodes-5-10_Samples-{num_samples_per_task}_Splits-1_Train.jsonl",
    }
    
    dataset_paths = dataset_path_templates
    
    # Task order for interleaving
    task_order = [
        "connectivity",
        "cycle_detection", 
        "degree",
        "isomorphism",
        "shortest_path",
        "max_common_subgraph",
        "max_clique",
    ]
    
    # Load all datasets
    print("Loading datasets...")
    task_datasets = {}
    for task_name in task_order:
        path = dataset_paths[task_name]
        try:
            ds = load_dataset("json", data_files=path, split="train")
            task_datasets[task_name] = ds
            print(f"Loaded {task_name}: {len(ds)} samples")
        except Exception as e:
            print(f"Warning: Failed to load {task_name} from {path}: {e}")
            continue
    
    if not task_datasets:
        raise ValueError("No datasets loaded successfully!")
    
    # Find the maximum dataset length
    max_length = max(len(ds) for ds in task_datasets.values())
    print(f"\nMaximum dataset length: {max_length}")
    
    # Interleave datasets in round-robin fashion
    print("\nInterleaving datasets...")
    interleaved_data = []
    
    for i in range(max_length):
        for task_name in task_order:
            if task_name in task_datasets:
                ds = task_datasets[task_name]
                if i < len(ds):
                    # Add the i-th sample from this task
                    interleaved_data.append(ds[i])
    
    print(f"Total interleaved samples: {len(interleaved_data)}")
    
    # Create a new dataset from the interleaved data
    # Get the features from the first dataset
    first_dataset = task_datasets[task_order[0]]
    interleaved_dataset = Dataset.from_dict({
        key: [sample[key] for sample in interleaved_data]
        for key in first_dataset.features.keys()
    })
    
    # Print statistics
    print("\nDataset statistics:")
    task_counts = {}
    for sample in interleaved_data:
        task = sample.get("task", "unknown")
        task_counts[task] = task_counts.get(task, 0) + 1
    
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count} samples")
    
    return interleaved_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Stage3 Joint GRPO training on multiple tasks")
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/GraphVulcan-SFT",
        help="Path to the SFT model from stage2 training"
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    # parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--num_generations", type=int, default=4, help="Number of completions to generate per prompt")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for generation")
    parser.add_argument("--beta", type=float, default=0.05, help="GRPO beta for preference strength")
    parser.add_argument("--num_samples_per_task", type=int, default=3000, help="Number of samples per task to load")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file")
    parser.add_argument("--report_to", type=str, default="none", choices=["tensorboard", "none"],
                        help="Logging platform: tensorboard or none")
    return parser.parse_args()

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    output_dir = f"{args.model_path}-s3-joint-GRPO-{timestamp}"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load interleaved joint dataset
    dataset = load_joint_dataset_interleaved(num_samples_per_task=args.num_samples_per_task)
    
    # Parse prompts from JSON strings to message lists using chat template
    # Pass tokenizer to ensure consistent formatting with inference
    print("\nParsing prompts...")
    dataset = dataset.map(lambda x: parse_prompt(x, tokenizer), num_proc=8)
    
    # Filter to only keep examples with "prompt" and "task" fields
    dataset = dataset.filter(lambda x: "prompt" in x and "task" in x)
    
    print(f"\nLoaded {len(dataset)} training samples after filtering")
    print("Sample:", dataset[0])
    
    # Store tasks for reward computation
    # Create a mapping from prompt to task
    prompt_to_task = {}
    for example in dataset:
        prompt_to_task[example["prompt"]] = example["task"]

    # Define reward function that will be called by GRPOTrainer
    # GRPOTrainer calls: reward_fn(completions, prompts=prompts, **kwargs)
    def reward_fn(completions: List[str], **kwargs) -> List[float]:
        # Extract tasks from prompts using the mapping
        prompts = kwargs.get("prompts", [])
        tasks = [prompt_to_task.get(p, "unknown") for p in prompts]
        # Add tasks to kwargs and pass everything to compute_reward
        kwargs["tasks"] = tasks
        rewards = compute_reward(completions, **kwargs)
        return rewards



    # Configure GRPO training
    if args.deepspeed is not None:
        print(f"Using DeepSpeed config: {args.deepspeed}")
        optim = "adamw_torch"
    else:
        print("DeepSpeed not specified, using 8-bit AdamW")
        optim = "paged_adamw_8bit"

    if args.report_to == "tensorboard":
        print("Setting up TensorBoard logging")
        logging_dir = os.path.join(output_dir, "logs")
        print(f"Logging directory: {logging_dir}")
    else:
        logging_dir = None

    training_args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=4,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_on_each_node=False,  # Only save on main node
        bf16=True,
        report_to=args.report_to,
        logging_dir=logging_dir,
        optim=optim,
        remove_unused_columns=False,
        save_only_model=True,
        # Disable shuffling to preserve interleaved task order
        shuffle_dataset=False,
        # max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        deepspeed=args.deepspeed,
        # DataLoader configuration to preserve order
        dataloader_drop_last=False,
        dataloader_num_workers=0,  # Set to 0 to ensure sequential loading
        dataloader_pin_memory=False,
        # GRPO-specific parameters
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=0.9,
        repetition_penalty=1.2,
        beta=args.beta,
        generation_kwargs={
            "max_new_tokens": args.max_completion_length,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "repetition_penalty": 1.2,
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": 0.9,
        }
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )

    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    # Start training
    print("\nStarting GRPO training...")
    trainer.train()

    # Save final model and tokenizer (only on main process)
    # In distributed training, only rank 0 should save the model
    # if trainer.is_world_process_zero():
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved at: {output_dir}")
    # else:
    #     print(f"Skipping model save on rank {trainer.args.local_rank}")
    
    
    # Print logging information
    if args.report_to == "tensorboard":
        print(f"\nTo view training metrics, run:")
        print(f"  tensorboard --logdir={logging_dir}")

if __name__ == "__main__":
    main()
