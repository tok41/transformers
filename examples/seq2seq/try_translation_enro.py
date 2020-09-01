import sys, os
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import AutoModelForSeq2SeqLM
from transformers import EncoderDecoderModel, BertTokenizer
import torch

def main():
    output_dir = 'enro_finetune_mbart01'
    model_dir = f'{output_dir}/best_tfmr'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    src_txt = ['Some of that money was allegedly funneled back to campaign coffers of the ruling party and its allies.', 
               'According to prosecutors, the scheme at Petrobras involved roughly $2 billion in bribes and other illegal funds.']
    for text in src_txt:
        print(f'src: {text}')
        input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
        generated = model.generate(input_ids)
        gen_txt = [tokenizer.decode(t, skip_special_tokens=True) for t in generated]
        print(f'trans: {gen_txt}')

if __name__ == '__main__':
    main()