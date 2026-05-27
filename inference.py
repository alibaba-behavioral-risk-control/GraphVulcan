import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
import argparse
import os
from datetime import datetime
from tqdm import tqdm
from evaluate.evaluate_connectivity import load_and_evaluate_s2_connectivity
from evaluate.evaluate_isomorphism import load_and_evaluate_s2_isomorphism
from evaluate.evaluate_degree import load_and_evaluate_s2_degree
from evaluate.evaluate_shortest_path import load_and_evaluate_s2_shortest_path
from evaluate.evaluate_max_common_subgraph import load_and_evaluate_s2_mcs
from evaluate.evaluate_cycle_detection import load_and_evaluate_s2_cycle_detection
from evaluate.evaluate_max_clique import load_and_evaluate_s2_max_clique
from graph_vocab.graph_vocabulary import GraphVocabulary
graph_vocab = GraphVocabulary()
timestamp_str = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

parser = argparse.ArgumentParser(description="LLM Inference for Graph Reasoning Tasks")
parser.add_argument("--test_data_path", type=str, default="data/s2_shortest_path/GraphVocab_Stage2_ShortestPath_CoT_Nodes-11-30_Samples-100_Splits-1_Test.jsonl", help="Path to test data JSON file")
parser.add_argument("--model_path", type=str, default="model/GraphVulcan-SFT", help="Path to the pretrained model")
parser.add_argument("--max_new_tokens", type=int, default=8192, help="Maximum number of tokens to generate")
parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
parser.add_argument("--task", type=str, default="s2_shortest_path", help="Select task")
parser.add_argument("--num_splits", type=int, default=10, help="Number of splits for evaluation")
parser.add_argument("--verbose", action="store_true", help="Print user messages and assistant responses during inference")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference (increase for faster multi-GPU inference)")
args = parser.parse_args()


def inference(test_data, tokenizer, model, output_path, verbose=True, batch_size=1):

    # Use pipeline with proper chat handling and batch processing
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        batch_size=batch_size  # Enable batch processing for faster inference
    )

    print(f"Starting inference with batch_size={batch_size}...")
    if not os.path.exists(os.path.dirname(os.path.abspath(output_path))):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Prepare all prompts first
    prompts = []
    valid_indices = []
    user_messages = []
    
    for idx, example in enumerate(test_data):
        messages = example.get("messages", [])
        if len(messages) < 2:
            print(f"Warning: Invalid format at index {idx}, skipping.")
            continue

        # Extract system and user messages
        system_msg = ""
        user_msg = ""
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            elif msg["role"] == "user":
                user_msg = msg["content"]
                break  # only first user message

        if not user_msg:
            print(f"Warning: No user message at index {idx}, skipping.")
            continue

        # Construct input using chat template if available
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            # Use official chat template
            input_messages = []
            if system_msg:
                input_messages.append({"role": "system", "content": system_msg})
            input_messages.append({"role": "user", "content": user_msg})
            prompt = tokenizer.apply_chat_template(
                input_messages,
                tokenize=False,
                add_generation_prompt=True  # adds assistant start token if supported
            )
        else:
            # Fallback: manual concatenation (adjust based on your model's expected format)
            if system_msg:
                prompt = f"{system_msg}\n\nUser: {user_msg}\n\nAssistant:"
            else:
                prompt = f"User: {user_msg}\n\nAssistant:"

        prompts.append(prompt)
        valid_indices.append(idx)
        user_messages.append(user_msg)

    # Process in batches
    total_samples = len(prompts)
    print(f"Total valid samples: {total_samples}")
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        # Process batches with progress bar
        for batch_start in tqdm(range(0, total_samples, batch_size), desc="Inference Progress", unit="batch"):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_prompts = prompts[batch_start:batch_end]
            batch_indices = valid_indices[batch_start:batch_end]
            batch_user_msgs = user_messages[batch_start:batch_end]
            
            if verbose and batch_size == 1:
                print(f"Processing test case {batch_indices[0]}...")
                print(f"User message: {batch_user_msgs[0]}")

            try:
                # Batch inference
                outputs = pipe(
                    batch_prompts,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_full_text=False  # Only return generated text, not the prompt
                )

                # Process each output in the batch
                for i, output in enumerate(outputs):
                    idx = batch_indices[i]
                    user_msg = batch_user_msgs[i]
                    prompt = batch_prompts[i]
                    
                    # Extract generated text
                    if isinstance(output, list):
                        generated_text = output[0]["generated_text"]
                    else:
                        generated_text = output["generated_text"]
                    
                    # Extract only the newly generated part (after the prompt)
                    if generated_text.startswith(prompt):
                        response = generated_text[len(prompt):].lstrip()
                    else:
                        response = generated_text
                    
                    if verbose and batch_size == 1:
                        print(f"Response: \n {response}")

                    # Build output in the exact same format as input
                    output_messages = []
                    output_messages.append({
                        "role": "user",
                        "content": user_msg.strip()
                    })
                    output_messages.append({
                        "role": "assistant",
                        "content": response.strip()
                    })

                    # Write as a single JSON object per line (JSONL)
                    result = {"messages": output_messages}
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                
                f_out.flush()

            except Exception as e:
                print(f"Error in batch starting at index {batch_start}: {e}")
                # Write error for each sample in the batch
                for i in range(len(batch_prompts)):
                    idx = batch_indices[i]
                    error_result = {
                        "id": idx,
                        "error": str(e),
                        "input_prompt": batch_prompts[i]
                    }
                    f_out.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                f_out.flush()

    print(f"Inference completed! Results saved to {output_path}")

if __name__ == "__main__":
    print(args)
    print("Loading test data...")
    test_data_path = f"data/{args.test_data_path}"
    model_path = f"model/{args.model_path}"
    test_data = load_dataset("json", data_files=test_data_path)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.verbose:
        for token in graph_vocab.GRAPH_TOKENS:
            encoded = tokenizer.tokenize(token)
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"{token} -> {encoded} -> {token_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    output_file_name = f"{args.model_path}/{args.test_data_path}"
    output_path = f"result/{output_file_name}"
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    inference(test_data, tokenizer, model, output_path, verbose=args.verbose, batch_size=args.batch_size)
    if args.task == "s2_connectivity":
        results = load_and_evaluate_s2_connectivity(output_path, tokenizer_obj=tokenizer, num_splits=args.num_splits, verbose=args.verbose)
    elif args.task == "s2_isomorphism":
        results = load_and_evaluate_s2_isomorphism(output_path, tokenizer_obj=tokenizer, num_splits=args.num_splits, verbose=args.verbose)
    elif args.task == "s2_degree":
        results = load_and_evaluate_s2_degree(output_path, tokenizer_obj=tokenizer, num_splits=args.num_splits, verbose=args.verbose)
    elif args.task == "s2_shortest_path":
        results = load_and_evaluate_s2_shortest_path(output_path, tokenizer_obj=tokenizer, num_splits=args.num_splits, verbose=args.verbose)
    elif args.task == "s2_max_common_subgraph":
        results = load_and_evaluate_s2_mcs(output_path, tokenizer_obj=tokenizer, num_splits=args.num_splits, verbose=args.verbose)
    elif args.task == "s2_cycle_detection":
        results = load_and_evaluate_s2_cycle_detection(output_path, tokenizer_obj=tokenizer, num_splits=args.num_splits, verbose=args.verbose)
    elif args.task == "s2_max_clique":
        results = load_and_evaluate_s2_max_clique(output_path, tokenizer_obj=tokenizer, num_splits=args.num_splits, verbose=args.verbose)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    







