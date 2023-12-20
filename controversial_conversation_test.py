import logging
import os
import os.path
from pathlib import Path
import json
import sys
import signal
import time
from datetime import datetime
from collections import OrderedDict
# import numpy as np
import torch

sys.path.append(os.path.abspath('/home/fmeng/workspace/llm_test'))
from generation import Llama
from model import Transformer, ModelArgs
from tokenizer import Tokenizer

from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)


g_llama2_model_dir = '/home/fmeng/workspace/llama_models'

# g_llama_model_path = '/home/fmeng/workspace/llama_models/llama-2-13b-chat'
g_tokenizer_path = '/home/fmeng/workspace/llama_models/tokenizer.model'
g_max_seq_len = 2048
g_top_p = 0.9
g_max_gen_len = None
g_max_batch_size = 4
g_temperature = 0.6

g_sys_msg = "Your responses are always relevant to the first user prompt. Be rigorous and honest on every sentence in responses. Always keep consistency and coherence in the entire dialog. Always explain why if you detect inconsistency or incoherence in the dialog. If you do not know the answer to a question, do not output false information or irrelevant information."

g_contro_prompt = """Your last response includes incorrect information. Explicitly identify incorrect statements, and correct them."""


def set_env_vars():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['LOCAL_RANK'] = '-1'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'


# def load_llama_model():
#     llama_model = Llama.build(
#         ckpt_dir=g_llama_model_path,
#         tokenizer_path=g_tokenizer_path,
#         max_seq_len=g_max_seq_len,
#         max_batch_size=g_max_batch_size,
#     )
#     return llama_model


def build_llama_model(model_id):
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("gloo")
    if not model_parallel_is_initialized():
        initialize_model_parallel(int(os.environ.get("WORLD_SIZE", 1)))

    with open(Path(g_llama2_model_dir) / model_id / "params.json", "r") as f:
        params = json.loads(f.read())
    model_args = ModelArgs(
        max_seq_len=g_max_seq_len,
        max_batch_size=g_max_batch_size,
        **params,
    )
    tokenizer = Tokenizer(model_path=g_tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    model = Transformer(model_args)
    state_dict = model.state_dict()

    if model_id == 'llama-2-13b-chat':
        num_chkpt = 2
        l_chkpt = []
        for i in range(num_chkpt):
            chkpt_i = torch.load(os.path.join(g_llama2_model_dir, model_id, 'consolidated.0%s.pth' % i),
                                 map_location="cpu")
            l_chkpt.append(chkpt_i)

    full_state_dict = OrderedDict.fromkeys(state_dict.keys())
    for weight_key in state_dict:
        if state_dict[weight_key].shape == l_chkpt[0][weight_key].shape:
            full_state_dict[weight_key] = l_chkpt[0][weight_key]
        elif state_dict[weight_key].shape[0] != l_chkpt[0][weight_key].shape[0]:
            full_state_dict[weight_key] = torch.cat([chkpt[weight_key] for chkpt in l_chkpt], dim=0)
        else:
            full_state_dict[weight_key] = torch.cat([chkpt[weight_key] for chkpt in l_chkpt], dim=1)

    model.load_state_dict(full_state_dict, strict=False)
    llama_model = Llama(model, tokenizer)
    return llama_model


def llama_chat(llama_model):
    def __sigint_handler(sig, frame):
        print('''\n[llama_chat] Quit Chat.''')
        print('------------------------------------------------------------\n')
        sys.exit(0)

    signal.signal(signal.SIGINT, __sigint_handler)

    if os.path.exists('TERM'):
        os.remove('TERM')

    # sys_msg = {
    #     "role": "system",
    #     "content": "Be rigorous and honest on every sentence in responses. Always keep consistency and coherence in the entire dialog. Always explain why if you detect inconsistency or incoherence in the dialog. If you do not know the answer to a question, do not output false information."
    # }



    print('You will start the conversation.\n')

    user_content = input('[%s You]: ' % 0)
    print('------------------------------------------------------------\n')
    user_msg = {'role': 'user', 'content': user_content}

    full_dialog = [user_msg]

    iter_idx = 0
    while True:
        if os.path.exists('TERM'):
            break

        start_t = time.time()
        response = llama_model.simple_chat_completion(
            full_dialog,
            sys_msg=g_sys_msg,
            max_gen_len=g_max_gen_len,
            temperature=g_temperature,
            top_p=g_top_p,
        )
        if response is None:
            logging.error('[llama_chat] Invalid response occurred. Quit.')
            break
        response_content = response[0]['generation']['content']
        logging.debug('[llama_chat] Iter: %s, Elapse: %s' % (iter_idx, time.time() - start_t))

        print('[%s Llama] %s' % (iter_idx, response_content))
        print('------------------------------------------------------------\n')

        llama_msg = {'role': response[0]['generation']['role'],
                     'content': response_content}
        full_dialog.append(llama_msg)
        iter_idx += 1

        user_content = g_contro_prompt
        user_msg = {'role': 'user', 'content': user_content}
        full_dialog.append(user_msg)
        print('[%s You] %s' % (iter_idx, user_content))
        print('------------------------------------------------------------\n')

    out_str = '\n\n'.join(['%s:%s:%s'
                           % (int((idx-1)/2) if idx % 2 != 0 and idx > 0 else int(idx/2),
                              msg['role'],
                              msg['content']) for idx, msg in enumerate(full_dialog)])
    with open('controversial_conversation_%s.txt' % datetime.strftime(datetime.now(), '%Y%m%d%H%M%S'), 'w+') as out_fd:
        out_fd.write(out_str)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # Force to use CPU
    set_env_vars()
    # Retrieve Llama model
    # llama_model = load_llama_model()
    llama_model = build_llama_model('llama-2-13b-chat')
    print(llama_model)

    torch.set_num_interop_threads(os.cpu_count())  # Inter-op parallelism
    torch.set_num_threads(os.cpu_count())  # Intra-op parallelism

    # Chat completion
    llama_chat(llama_model)


