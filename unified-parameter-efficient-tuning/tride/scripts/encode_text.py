"""this script encodes the given text piece,
potentially allowing for different hidden
representations
"""

import os
import json
import shutil
import numpy as np
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

def encode_none(args, save_dir, ids):
    encodings = tokenizer('\n'.join(open(args.data).readlines()), return_tensors='pt')

    keys = np.memmap(os.path.join(save_dir, f'keys.{ids}.size{encodings.input_ids.size(1)}.hid{model.config.n_embd}.npy'),
                     dtype=np.float32,
                     mode='w+',
                     shape=(encodings.input_ids.size(1), model.config.n_embd))
    vals = open(os.path.join(save_dir, f'vals.{ids}.jsonl'), 'w')
    max_length = model.config.n_positions

    stride = 512
    lls = []
    cur = 0
    # import pdb; pdb.set_trace()
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            # labels are shifted inside the model
            outputs = model(input_ids, labels=target_ids,
                output_hidden_states=True, return_hidden_type=args.return_hidden_type)

            # note that the actual number tokens to compute the loss is (trg_len -1)
            # here the ppl computation is incorrect
            log_likelihood = outputs[0] * trg_len
            hidden_states = outputs.hidden_states

            hidden_states = hidden_states[args.nlayer]
            keys[cur:cur+trg_len] = hidden_states[0][-trg_len:, :].cpu()
            toks = tokenizer.convert_ids_to_tokens(input_ids[0, -trg_len:].tolist())
            toks = [tokenizer.convert_tokens_to_string([t]) for t in toks]

            for t in toks:
                vals.write(json.dumps(t))
                vals.write('\n')
            cur += trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    print(f'perplexity {ppl}')
    vals.close()

def encode_eol(args, save_dir, ids):
    lls = []
    numx = 0
    total_tok = 0
    skip = 0

    bx = []

    # import pdb; pdb.set_trace()
    data_file = open(args.data)

    for d, line in tqdm(enumerate(data_file)):
        if line.strip() == '':
            continue
            skip += 1
        encodings = tokenizer(line, return_tensors='pt')
        keys = np.memmap(os.path.join(save_dir, f'keys.id{d}.{ids}.size{encodings.input_ids.size(1)}.hid{model.config.n_embd}.npy'),
                         dtype=np.float32,
                         mode='w+',
                         shape=(encodings.input_ids.size(1), model.config.n_embd))
        vals = open(os.path.join(save_dir, f'vals.id{d}.{ids}.jsonl'), 'w')

        input_ids = encodings.input_ids.to(device)
        target_ids = input_ids.clone()
        trg_len = input_ids.size(1)

    # the perplexity computation is not very accurate
    # since it ignores the first token prediction and eos prediction

        with torch.no_grad():
            # labels are shifted inside the model
            outputs = model(input_ids, labels=target_ids,
                output_hidden_states=True, return_hidden_type=args.return_hidden_type)
            log_likelihood = outputs[0] * (trg_len-1)
            total_tok += trg_len-1
            hidden_states = outputs.hidden_states

            # import pdb; pdb.set_trace()
            hidden_states = hidden_states[args.nlayer]

            assert trg_len == hidden_states.size(1)

            keys[:] = hidden_states[0][:,:].cpu()
            toks = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            toks = [tokenizer.convert_tokens_to_string([t]) for t in toks]

            for t in toks:
                vals.write(json.dumps(t))
                vals.write('\n')
                    # keys[cur+i] = hidden_states[i, 1, :].cpu().numpy().astype(np.float32)

        lls.append(log_likelihood)

        vals.close()
    
    print(f'skipped {skip} lines')
    ppl = torch.exp(torch.stack(lls).sum() / total_tok)
    print(f'perplexity {ppl}')




parser = argparse.ArgumentParser()
parser.add_argument('data', type=str,
    help='the input text file')
parser.add_argument('--nlayer', type=int, default=-1,
    help='the order of layer from which we extract hidden states, \
    default to the last layer')
parser.add_argument('--save-emb', type=str, default='hidden_val',
    help='the folder to save the encodings')
parser.add_argument('--model', type=str, default='gpt2-large',
    help='the pretrained model name')
parser.add_argument('--return-hidden-type', type=str, default=None, \
    choices=['ffn_input_after_ln', 'standard'], \
    help='the hidden representations to use, by default we use the output of every \
    sub transformer layer')
parser.add_argument('--break-mode', type=str, default='none', \
    choices=['none', 'eol'], \
    help='none means no break, eol is end of line')

args = parser.parse_args()

save_dir = f'{args.save_emb}'

# if os.path.isdir(save_dir):
#     shutil.rmtree(save_dir)


os.makedirs(args.save_emb, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model)

model.to(device)
model.eval()

hidden_type = 'standard' if args.return_hidden_type is None else args.return_hidden_type
ids = f'layer{args.nlayer}.rpr_type_{hidden_type}'

if args.break_mode == 'none':
    encode_none(args, save_dir, ids)
elif args.break_mode == 'eol':
    encode_eol(args, save_dir, ids)
else:
    raise ValueError
