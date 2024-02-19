import fire
import torch

from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# EXAMPLE USAGE:
# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 llama_arXiv/interactive_chatbot_finetuned.py
# --ckpt_dir llama_arXiv/results_arxiv/final_checkpoint --base_model_name Llama-2-7b-chat-hf --max_seq_len 512
# --max_batch_size 6



def main(
    ckpt_dir: str,
    base_model_name: str,
    max_new_tokens: int = 1000,
    local_rank=None
):
    """
    Entry point of the program for generating text using a pretrained model.
    ...
    """
    device_map = {"": 0}
    model = AutoPeftModelForCausalLM.from_pretrained(
        ckpt_dir, device_map=device_map, torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print('You can now enter your input:')
    while True:
        user_input = input()
        if user_input.lower() == 'exit':
            break
        
        inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
        outputs = model.generate(
            input_ids=inputs["input_ids"].to("cuda"),
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)
