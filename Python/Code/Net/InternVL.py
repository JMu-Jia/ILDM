import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

def run(args):
    print(torch.bfloat16)
    model = AutoModel.from_pretrained(
        '/root/data1/Research/SCT/Python/ExCode/InternVL/InternVL',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).cuda().eval()

    print('end')