import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List

import src.scaling_utils


raw_data_dir = "data/raw_data/pretraining_causal_language_modeling"
os.makedirs(raw_data_dir, exist_ok=True)

max_context_length = 2048
model_nicknames_to_huggingface_paths_dict = {
    "Cerebras_111M_2B": "cerebras/Cerebras-GPT-111M",
    "Cerebras_256M_5B": "cerebras/Cerebras-GPT-256M",
    "Cerebras_590M_11B": "cerebras/Cerebras-GPT-590M",
    "Cerebras_1.3B_26B": "cerebras/Cerebras-GPT-1.3B",
    "Cerebras_2.7B_54B": "cerebras/Cerebras-GPT-2.7B",
    "Cerebras_6.7B_134B": "cerebras/Cerebras-GPT-6.7B",
    "Cerebras_13B_260B": "cerebras/Cerebras-GPT-13B",
}

dataset_dict = {
    # "c4": "allenai/c4",
    "Zyda-2": "Zyphra/Zyda-2",
    # "The Pile": "monology/pile-test-val",
    # "MiniPile": "JeanKaddour/minipile",
    # "LAMBADA": "EleutherAI/lambada_openai",
    # "RedPajama": "togethercomputer/RedPajama-Data-1T-Sample",
    # "Fineweb": "HuggingFaceFW/fineweb",
}

for dataset_name, dataset_hf_path in dataset_dict.items():
    try:
        sequences: List[
            str
        ] = src.scaling_utils.prepare_pretraining_causal_language_modeling_dataset(
            dataset_hf_path=dataset_hf_path,
        )
    except Exception as e:
        print(f"Error: {e}")
        continue

    for (
        model_nickname,
        huggingface_path,
    ) in model_nicknames_to_huggingface_paths_dict.items():
        tokenizer = AutoTokenizer.from_pretrained(
            huggingface_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            huggingface_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

        log_probs_dict = {
            "log_probs": [],
            "Problem Idx": [],
            "Seq Idx": [],
        }
        for sample_idx, sequence in enumerate(sequences[:1000]):
            # for sample_idx, sequence in enumerate(sequences[:10]):
            encoded_sequence = tokenizer(
                sequence,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=max_context_length,
            ).to(model.device)

            with torch.no_grad():
                input_ids = encoded_sequence.input_ids
                labels = input_ids.clone()
                labels = labels[:, 1:]  # Remove first position

                output = model(**encoded_sequence)

                logits = output.logits
                logits = logits[
                    :, :-1, :
                ]  # Remove last position as it has nothing to predict
                # Apply log softmax over vocabulary dimension
                log_probs = torch.log_softmax(logits, dim=-1)
                # Gather log probs for the actual tokens in the sequence.
                # Shape: (sequence_length,)
                token_log_probs = torch.gather(
                    log_probs, 2, labels.unsqueeze(-1)
                ).squeeze()

            sequence_indices = torch.arange(
                token_log_probs.size(0), device=token_log_probs.device
            )
            log_probs_dict["log_probs"].extend(token_log_probs.cpu().numpy().tolist())
            log_probs_dict["Problem Idx"].extend([sample_idx] * token_log_probs.size(0))
            log_probs_dict["Seq Idx"].extend(sequence_indices.cpu().numpy().tolist())

        log_probs_df = pd.DataFrame.from_dict(log_probs_dict)
        log_probs_df["Model Nickname"] = model_nickname
        log_probs_df["Dataset"] = dataset_name

        log_probs_df.to_parquet(
            os.path.join(
                raw_data_dir,
                f"{dataset_name}-{model_nickname}-log_probs.parquet",
            ),
            index=False,
        )
        print(f"Saved log probs for {model_nickname} on {dataset_name}.")

        del model, tokenizer, log_probs_dict, log_probs_df
