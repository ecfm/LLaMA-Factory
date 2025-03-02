#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import gc
import json
import logging
import time
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import GenerationConfig

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model, load_tokenizer


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('inference.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned LLM model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="export_model",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/testing_data_s2n_short_v0.json",
        help="Path to the test file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="inference_results.json",
        help="Path to save the results"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="qwen",
        help="Template to use for formatting prompts"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of items to process in parallel (default: 8 for A100 GPU)"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save results every N batches"
    )
    return parser.parse_args()


def load_test_data(file_path: str) -> List[Dict[str, Any]]:
    """Load the test data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def group_by_length(items, batch_size):
    """Group items by similar input length to reduce padding overhead."""
    # Extract prompt length for each item
    items_with_length = []

    for item in items:
        # Find the human message
        human_message = ""
        for msg in item["conversations"]:
            if msg["from"] == "human":
                human_message = msg["value"]
                break

        items_with_length.append((item, len(human_message)))

    # Sort by length
    items_with_length.sort(key=lambda x: x[1])

    # Group into batches
    grouped_items = []
    for i in range(0, len(items_with_length), batch_size):
        batch = [item[0] for item in items_with_length[i:i+batch_size]]
        grouped_items.append(batch)

    return grouped_items

def process_batch(encoded_inputs, tokenizer, model, is_chatml_format, assistant_start):
    """Process a batch for inference."""
    # Create a padded batch tensor
    padded_input_ids = []
    padded_attention_masks = []

    for item in encoded_inputs:
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.long)

        padded_input_ids.append(input_ids)
        padded_attention_masks.append(attention_mask)

    # Find max length for padding
    max_length = max(len(x) for x in padded_input_ids)

    # Pad all tensors to max length
    for i in range(len(padded_input_ids)):
        input_ids = padded_input_ids[i]
        attention_mask = padded_attention_masks[i]

        # Calculate padding needed
        padding_length = max_length - len(input_ids)

        if padding_length > 0:
            # Create padding tensors with correct dtype
            input_padding = torch.full((padding_length,), tokenizer.pad_token_id,
                                      dtype=torch.long, device=input_ids.device)
            mask_padding = torch.zeros(padding_length, dtype=torch.long, device=attention_mask.device)

            # Left padding for generation
            padded_input_ids[i] = torch.cat([input_padding, input_ids])
            padded_attention_masks[i] = torch.cat([mask_padding, attention_mask])

    # Stack into batch tensors
    batch_input = {
        "input_ids": torch.stack(padded_input_ids).to(model.device),
        "attention_mask": torch.stack(padded_attention_masks).to(model.device)
    }

    # Add assistant tokens for ChatML if needed
    if is_chatml_format and len(assistant_start) > 0:
        try:
            # Convert to tensor once for efficiency - make sure it's a Long tensor
            assistant_tensor = torch.tensor(assistant_start, dtype=torch.long, device=model.device)

            # Apply to each input in the batch
            modified_inputs = []
            modified_attention_masks = []

            for i in range(len(batch_input["input_ids"])):
                # Get current tensor
                current_input = batch_input["input_ids"][i]
                current_mask = batch_input["attention_mask"][i]

                # Concatenate along dimension 0
                new_input = torch.cat([current_input, assistant_tensor])
                new_mask = torch.cat([current_mask, torch.ones(len(assistant_start),
                                                             dtype=current_mask.dtype,
                                                             device=model.device)])

                modified_inputs.append(new_input)
                modified_attention_masks.append(new_mask)

            # Stack tensors back into a batch
            max_length = max(len(x) for x in modified_inputs)

            # Pad tensors to max length
            padded_inputs = []
            padded_masks = []

            for inp, mask in zip(modified_inputs, modified_attention_masks):
                if len(inp) < max_length:
                    # Create padding - ensure we use the right dtype
                    padding = torch.full((max_length - len(inp),),
                                        tokenizer.pad_token_id,
                                        dtype=torch.long,
                                        device=inp.device)
                    mask_padding = torch.zeros(max_length - len(mask),
                                             dtype=mask.dtype,
                                             device=mask.device)

                    # Pad the tensors
                    padded_inp = torch.cat([padding, inp])  # Left padding
                    padded_mask = torch.cat([mask_padding, mask])  # Left padding

                    padded_inputs.append(padded_inp)
                    padded_masks.append(padded_mask)
                else:
                    padded_inputs.append(inp)
                    padded_masks.append(mask)

            # Stack into batch tensors
            batch_input["input_ids"] = torch.stack(padded_inputs)
            batch_input["attention_mask"] = torch.stack(padded_masks)
        except Exception as e:
            logger.error(f"Error adding assistant start token: {e}")

    # Ensure input_ids are long tensors for embedding layer
    if batch_input["input_ids"].dtype != torch.long:
        logger.warning(f"Converting input_ids from {batch_input['input_ids'].dtype} to torch.long")
        batch_input["input_ids"] = batch_input["input_ids"].to(dtype=torch.long)

    return batch_input

def process_output(output, encoded_input, item_id, human_message, target_message, attention_mask,
                  tokenizer, generating_args, is_chatml_format, assistant_start):
    """Process a single output."""
    # Find where the prompt ends and the generated text begins
    input_length = len(encoded_input["input_ids"])

    # For left-padded sequences, we need to calculate the actual starting position
    # based on the attention mask
    original_attention_mask = attention_mask.cpu().tolist()

    # Find the real beginning of the sequence (after padding)
    padding_length = 0
    for mask_val in original_attention_mask:
        if mask_val == 0:
            padding_length += 1
        else:
            break

    # Calculate the true prompt length including assistant tokens but excluding padding
    if is_chatml_format and len(assistant_start) > 0:
        prompt_length = input_length + len(assistant_start) + padding_length
    else:
        prompt_length = input_length + padding_length

    # Get the generated part (excluding prompt)
    generated_ids = output[prompt_length:]

    # Decode the generated text
    response_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=generating_args.to_dict().get("skip_special_tokens", True),
        clean_up_tokenization_spaces=True
    )

    # Final cleanup - trim any leading/trailing whitespace
    response_text = response_text.strip()

    # Save the result with target information
    result = {
        "id": item_id,
        "input": human_message,
        "output": response_text,
        "target": target_message,
        "prompt_length": input_length
    }

    return result

def run_inference():
    """Run inference on the test data with the fine-tuned model using true batching."""
    args = parse_args()
    start_time = time.time()

    logger.info("Optimizing for high-performance single-GPU inference with true batching")

    # Enable CUDA optimizations
    if torch.cuda.is_available():
        # Enable TF32 for faster matrix operations (A100 GPUs)
        # torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.allow_tf32 = True
        # Enable benchmarking to optimize CUDA kernels
        torch.backends.cudnn.benchmark = True
        # Enable memory-efficient attention if available
        # torch.backends.cuda.enable_mem_efficient_sdp = True
        # Enable flash attention if available
        torch.backends.cuda.enable_flash_sdp = True

        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Setup model with single-GPU optimizations
    model_args_dict = {
        "model_name_or_path": args.model_path,
        "trust_remote_code": True,
        "finetuning_type": "full",
        "infer_backend": "huggingface",
        "infer_dtype": "bfloat16",
        "flash_attn": "fa2",  # Will use if available
        "use_cache": True,    # Essential for faster generation
    }

    data_args_dict = {
        "template": args.template,
        "cutoff_len": 512,
    }

    finetuning_args_dict = {
        "stage": "sft",
    }

    # Optimized generation parameters for faster inference
    generating_args_dict = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "repetition_penalty": 1.2,
        "do_sample": args.temperature > 0.01,  # Use greedy decoding if temp is very low
        "num_beams": 1,  # Ensure greedy decoding for maximum speed
        "skip_special_tokens": True, # Skip special tokens like <|im_start|>, <|im_end|>
    }

    # Create the combined args dictionary
    all_args = {
        **model_args_dict,
        **data_args_dict,
        **generating_args_dict,
        **finetuning_args_dict,
    }

    # Initialize the model and tokenizer
    model_args, data_args, finetuning_args, generating_args = get_infer_args(all_args)

    # Clear CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading tokenizer...")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Set padding side to left for generation
    tokenizer.padding_side = "left"

    # Load the model with optimized settings
    print("Loading model with optimized settings...")
    model = load_model(tokenizer, model_args, finetuning_args)

    # Apply torch.compile for faster inference if available (PyTorch 2.0+)
    if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
        print("Applying torch.compile for faster inference...")
        try:
            # Try max-autotune mode first (most aggressive)
            model = torch.compile(model, mode="max-autotune", fullgraph=True)
            print("Model successfully compiled with max-autotune!")
        except Exception as e:
            print(f"Max-autotune compile failed, trying reduce-overhead: {e}")
            try:
                # Fall back to reduce-overhead if max-autotune fails
                model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                print("Model successfully compiled with reduce-overhead!")
            except Exception as e:
                print(f"Torch compile failed, falling back to standard model: {e}")

    # Identify assistant prefixes to strip based on the template
    assistant_prefixes = []
    for slot in template.format_assistant.slots:
        if isinstance(slot, str) and "{{content}}" not in slot:
            assistant_prefixes.append(slot.strip())

    # Identify assistant suffixes to strip based on the template
    assistant_suffixes = []
    for stop_word in template.stop_words:
        assistant_suffixes.append(stop_word)

    # For Qwen/ChatML templates, check for special prefixes
    is_chatml_format = False
    if data_args.template == "qwen" or data_args.template == "chatml":
        is_chatml_format = True
        print("Using Qwen/ChatML template for formatting")

    # Load test data
    test_data = load_test_data(args.test_file)
    print(f"Loaded {len(test_data)} test examples")

    # Group items by similar length for more efficient batching
    batch_size = args.batch_size
    batches = group_by_length(test_data, batch_size)
    print(f"Grouped into {len(batches)} optimized batches")

    # Run inference and save results
    results = []
    total_items = 0

    # Track performance metrics
    batch_times = []

    # Process in optimized batches
    for batch_idx, batch_items in enumerate(tqdm(batches, desc=f"Processing batches (size ~{batch_size})")):
        try:
            batch_start = time.time()

            # Prepare the messages for each item
            item_ids = []
            human_messages = []
            target_messages = []
            input_messages = []

            for item in batch_items:
                item_id = item["id"]
                item_ids.append(item_id)

                # Get the human message (input) and target message
                human_message = None
                target_message = ""

                for msg in item["conversations"]:
                    if msg["from"] == "human":
                        human_message = msg["value"]
                    elif msg["from"] == "gpt":
                        target_message = msg["value"]

                if not human_message:
                    print(f"Warning: No human message found for item {item_id}")
                    continue

                human_messages.append(human_message)
                target_messages.append(target_message)

                # Create conversation with ONLY the user message
                input_message = [
                    {"role": "user", "content": human_message}
                ]

                input_messages.append(input_message)

            # Skip empty batches
            if not input_messages:
                logger.warning(f"Skipping empty batch {batch_idx}")
                continue

            # Encode all inputs in the batch
            encoded_inputs = []
            for idx, messages in enumerate(input_messages):
                try:
                    # Get system message if available
                    system = template.default_system

                    # Encode the input using the template
                    prompt_ids, _ = template.encode_oneturn(tokenizer, messages+[{"role": "assistant", "content": ""}], system)

                    # Store the encoded input
                    encoded_inputs.append({
                        "input_ids": prompt_ids,
                        "attention_mask": torch.tensor([1] * len(prompt_ids), dtype=torch.long)
                    })
                except Exception as e:
                    logger.error(f"Error encoding input {idx} in batch {batch_idx}: {e}")

            # Skip if we couldn't encode any inputs
            if not encoded_inputs:
                logger.warning(f"Skipping batch {batch_idx} as no inputs could be encoded")
                continue

            # Run inference with the model
            results_for_batch = []

            # Initialize assistant_start for ChatML format
            assistant_start = []
            if is_chatml_format:
                try:
                    # For Qwen/ChatML, we need to add the assistant start token for proper completion
                    assistant_start = tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
                except Exception as e:
                    logger.error(f"Error initializing assistant start token: {e}")
                    assistant_start = []

            # Process batch for inference
            batch_input = process_batch(encoded_inputs, tokenizer, model, is_chatml_format, assistant_start)

            with torch.inference_mode():
                # Get the template's stop tokens for proper generation termination
                stop_token_ids = template.get_stop_token_ids(tokenizer)

                # Create a proper generation config
                gen_config = GenerationConfig(**generating_args.to_dict())

                # Run the generation
                outputs = model.generate(
                    **batch_input,
                    generation_config=gen_config,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=stop_token_ids
                )

                # Process outputs
                for i, output in enumerate(outputs):
                    if i >= len(item_ids):  # Safety check
                        continue

                    # Process output
                    result = process_output(
                        output,
                        encoded_inputs[i],
                        item_ids[i],
                        human_messages[i],
                        target_messages[i],
                        batch_input["attention_mask"][i],
                        tokenizer,
                        generating_args,
                        is_chatml_format,
                        assistant_start
                    )
                    results_for_batch.append(result)

            # Add all results from this batch
            results.extend(results_for_batch)

            # Track performance
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            total_items += len(batch_items)  # Only processing each item once now

            # Save results periodically
            if (batch_idx + 1) % args.save_every == 0:
                with open(args.output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(results)} results so far")

                # Performance metrics
                avg_time_per_batch = np.mean(batch_times)
                avg_time_per_item = avg_time_per_batch / batch_size
                items_per_second = total_items / (batch_end - start_time)
                print(f"Performance: {items_per_second:.2f} items/sec, {avg_time_per_item:.2f} sec/item")

            # Clear CUDA cache periodically to prevent memory fragmentation
            if (batch_idx + 1) % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")

    # Save final results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Final performance report
    end_time = time.time()
    total_time = end_time - start_time
    items_per_second = len(results) / total_time

    print(f"Inference completed in {total_time:.2f} seconds")
    print(f"Processed {len(results)} items at {items_per_second:.2f} items/second")
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    run_inference()
