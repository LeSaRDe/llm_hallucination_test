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
import numpy as np
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
g_tokenizer_path = '/home/fmeng/workspace/llama_models/tokenizer.model'
g_max_seq_len = 2048
g_top_p = 0.9
g_max_gen_len = None
g_max_batch_size = 4
g_temperature = 1

# g_sys_msg = "Your responses are always relevant to the first user prompt. Be rigorous and honest on every sentence in responses. Always keep consistency and coherence in the entire dialog. Always explain why if you detect inconsistency or incoherence in the dialog. If you do not know the answer to a question, do not output false information or irrelevant information."

g_sys_msg = None

g_contro_prompt = """Your last response includes incorrect information. Explicitly identify incorrect statements, and correct them."""

# g_more_prompt = """More."""
g_more_prompt = """More about Computer Science?"""

g_l_rand_prompt = [
    """Is Abraham Lincoln a president of the United States?""",
    """Does gravity exist on the Earth?""",
    """Is Jazz the best music in the world?""",
    """What is the best coffee bean?""",
    """Is O(nlogn) the theoretical time lower bound of sorting algorithms?""",
    """Is Henri Matisse more talented than Pablo Picasso?""",
    """Can H2O be toxic to humans?""",
    """Why do we have to take flu shot every every year?""",
    """Is cheese cake better than pizza?""",
    """Is Calculus a requirement for machine learning?""",
    """What is the average salary of NBA players?""",
    """Does Orca breath in the water?""",
    """Is time travel actually possible?""",
    """What are differences between Finance and Economics?""",
    """Is the following sentence a proposition: this sentence is false?""",
    """Why are jury members not required to be professional in law?""",
    """Was Christian originated from Judaism?""",
    """Is philosophy practically useful in solving real-world problems?""",
    """Does earthquake always lead to tsunami?""",
    """Can conformity actually be controlled by government?"""
]

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

    if model_id == 'llama-2-13b-chat' or model_id == 'llama-2-13b':
        num_chkpt = 2
    elif model_id == 'llama-2-7b-chat' or model_id == 'llama-2-7b':
        num_chkpt = 1
    elif model_id == 'llama-2-70b-chat' or model_id == 'llama-2-70b':
        num_chkpt = 8

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


def get_user_prompt(p_type):
    if p_type == 'fixed_contro':
        return g_contro_prompt
    elif p_type == 'more':
        return g_more_prompt
    elif p_type == 'random':
        return g_l_rand_prompt[np.random.choice(len(g_l_rand_prompt))]

def llama_chat(llama_model, p_type='fixed_contro', history='full'):
    run_id = 'chat_%s' % datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    def __sigint_handler(sig, frame):
        print('''\n[llama_chat] Quit Chat.''')
        print('------------------------------------------------------------\n')
        sys.exit(0)

    signal.signal(signal.SIGINT, __sigint_handler)

    if os.path.exists('TERM'):
        os.remove('TERM')

    print('You will start the conversation.\n')

    if p_type == 'fixed_contro' or p_type == 'more':
        user_content = input('[%s You]: ' % 0)
    elif p_type == 'random':
        user_content = get_user_prompt(p_type)
    else:
        logging.error('[llama_chat] Invalid prompt type %s. Quit.' % p_type)
        sys.exit(-1)

    print('------------------------------------------------------------\n')
    init_msg = {'role': 'user', 'content': user_content}
    full_dialog = [init_msg]

    full_h_seq = []

    iter_idx = 0
    while True:
        if os.path.exists('TERM'):
            break

        start_t = time.time()
        if p_type == 'fixed_contro' or p_type == 'more':
            if history == 'full':
                in_dialog = full_dialog
            elif history == 'last':
                if len(full_dialog) >= 5:
                    in_dialog = [full_dialog[0], # init prompt
                                 full_dialog[1], # first Llama response
                                 full_dialog[-3], # last contro prompt
                                 full_dialog[-2], # last Llama response to last contro prompt
                                 full_dialog[-1], # new contro prompt
                                 ]
                else:
                    in_dialog = full_dialog
            else:
                logging.error('[llama_chat] Invalid history parameter: %s. Quit.' % history)
                sys.exit(-1)
        else:
            in_dialog = [full_dialog[-1]]
        response, actual_prompt, l_h_seq = llama_model.simple_chat_completion(
            in_dialog,
            sys_msg=g_sys_msg,
            max_gen_len=g_max_gen_len,
            temperature=g_temperature,
            top_p=g_top_p,
            keep_init_prompt=True
        )
        if response is None:
            logging.error('[llama_chat] Invalid response occurred. Quit.')
            break
        response_content = response[0]['generation']['content']
        # logging.debug('[llama_chat] Iter: %s, Elapse: %s' % (iter_idx, time.time() - start_t))

        print('[%s Prompt] %s\n' % (iter_idx, actual_prompt))

        print('[%s Llama] (Elapse: %s) %s' % (iter_idx, np.round(time.time() - start_t, decimals=2), response_content))
        print('------------------------------------------------------------\n')

        llama_msg = {'role': response[0]['generation']['role'],
                     'content': response_content}
        full_dialog.append(llama_msg)
        iter_idx += 1
        user_content = get_user_prompt(p_type=p_type)
        user_msg = {'role': 'user', 'content': user_content}
        full_dialog.append(user_msg)
        # print('[%s You] %s' % (iter_idx, user_content))
        # print('------------------------------------------------------------\n')

        full_h_seq.append([iter_idx, l_h_seq])

    out_str = '\n\n'.join(['%s:%s:%s'
                           % (int((idx-1)/2) if idx % 2 != 0 and idx > 0 else int(idx/2),
                              msg['role'],
                              msg['content']) for idx, msg in enumerate(full_dialog)])
    with open('%s_%s.txt' % (run_id, 'log'), 'w+') as out_fd:
        out_fd.write(out_str)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Model choice
    model_choice = input('Please choose the model to be used:\n'
                     '[1] llama-2-7b-chat\n'
                     '[2] llama-2-7b\n'
                     '[3] llama-2-13b-chat\n'
                     '[4] llama-2-13b\n'
                     '[5] llama-2-70b-chat\n'
                     '[6] llama-2-70b\n'
                     'Your choice: ')
    print('\n')

    # Prompt type
    p_type = input('Please choose the prompt type:\n'
                   '[1] Controversial Dialog\n'
                   '[2] More\n'
                   '[3] Random Prompt\n'
                   'Your choice: ')
    print('\n')

    model_choice = int(model_choice)
    if model_choice == 1:
        model_id = 'llama-2-7b-chat'
    elif model_choice == 2:
        model_id = 'llama-2-7b'
    elif model_choice == 3:
        model_id = 'llama-2-13b-chat'
    elif model_choice == 4:
        model_id = 'llama-2-13b'
    elif model_choice == 5:
        model_id = 'llama-2-70b-chat'
    elif model_choice == 6:
        model_id = 'llama-2-70b'
    else:
        logging.error('Your choice is not valid.')
        sys.exit(-1)

    # Force to use CPU
    set_env_vars()
    # Retrieve Llama model
    llama_model = build_llama_model(model_id)

    # Activate full threading
    torch.set_num_interop_threads(os.cpu_count())  # Inter-op parallelism
    torch.set_num_threads(os.cpu_count())  # Intra-op parallelism

    # Chat completion
    p_type = int(p_type)
    if p_type == 1:
        llama_chat(llama_model, p_type='fixed_contro', history='full')
    elif p_type == 2:
        llama_chat(llama_model, p_type='more', history='full')
    elif p_type == 3:
        llama_chat(llama_model, p_type='random')
    else:
        logging.error('Your choice is not valid.')
        sys.exit(-1)


