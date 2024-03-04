# Filename: gpt-neo-2.7b-generation.py
import os
import deepspeed
import torch
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM

local_rank = int(os.getenv('LOCAL_RANK', '3'))
world_size = int(os.getenv('WORLD_SIZE', '3'))
#generator = pipeline('text-generation', model='facebook/opt-6.7b',
#        )#device_map="auto")#.eval()#=local_rank)
tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-30b-hf')
model = AutoModelForCausalLM.from_pretrained(
        'decapoda-research/llama-30b-hf',
        
    ).eval()
print(f"model is loaded on device {model.device.type}")

def generate(tokenizer, model, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda")

    # fails here
    completion = model.generate(
        inputs.input_ids,
        do_sample=True, max_length=100
    )

    return completion

ds_model = deepspeed.init_inference(model,
                                           mp_size=4,#world_size,
                                          # dtype=torch.half,
                                           replace_method="auto",
                                           replace_with_kernel_inject=True)
with torch.no_grad():
    string=generate(tokenizer, ds_model, "Deepspeed is")
    #string = generator("DeepSpeed is", do_sample=True)#, min_length=50)
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(string)

with torch.no_grad():
    string=generate(tokenizer, ds_model, "Deepspeed is bitch")
    #string = generator("DeepSpeed is", do_sample=True)#, min_length=50)
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(string)
