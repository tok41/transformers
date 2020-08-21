import sys, os
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import AutoModelForSeq2SeqLM

def main():
    output_dir = 'jaja_finetune_full'
    model_dir = f'{output_dir}/best_tfmr'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    src_txt = ['amazon', 'anazon', 'amazom', 'やほー']
    print(f'src: {src_txt}')
    #tokenized_txt = tokenizer.tokenize(src_txt)
    #print(f'tokn: {tokenized_txt}')

    translated = model.generate(**tokenizer.prepare_translation_batch(src_txt))
    trs_txt = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    print(f'trans: {trs_txt}')

if __name__ == '__main__':
    main()