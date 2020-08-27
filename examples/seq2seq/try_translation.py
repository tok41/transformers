import sys, os
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import AutoModelForSeq2SeqLM
from transformers import EncoderDecoderModel, BertTokenizer
import torch

def main():
    output_dir = 'jaja_finetune_bert02'
    model_dir = f'{output_dir}/best_tfmr'
    #model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    #tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = EncoderDecoderModel.from_pretrained(model_dir)

    #src_txt = ['amazon', 'anazon', 'amazom', 'やほー', '私はAnazonで本を買いました。']
    #print(f'src: {src_txt}')
    #tokenized_txt = tokenizer.tokenize(src_txt)
    #print(f'tokn: {tokenized_txt}')

    #translated = model.generate(**tokenizer.prepare_translation_batch(src_txt))
    #trs_txt = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    #print(f'trans: {trs_txt}')

    # EncDecモデル
    src_txt = ['amazon', 'anazon', 'amazom', 'やほー', '私はAnazonで本を買いました。']
    for text in src_txt:
        print(f'src: {text}')
        input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
        generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)
        gen_txt = [tokenizer.decode(t, skip_special_tokens=True) for t in generated]
        print(f'trans: {gen_txt}')

if __name__ == '__main__':
    main()