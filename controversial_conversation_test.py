import logging
import os
import os.path
import sys
import signal
import time
from datetime import datetime
# import numpy as np

sys.path.append(os.path.abspath('/home/fmeng/workspace/llm_test'))
from generation import Llama


g_llama_model_path = '/home/fmeng/workspace/llama_models/llama-2-7b-chat'
g_tokenizer_path = '/home/fmeng/workspace/llama_models/tokenizer.model'
g_max_seq_len = 4096
g_top_p = 0.9
g_max_gen_len = None
g_max_batch_size = 4
g_temperature = 0.6


def set_env_vars():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['LOCAL_RANK'] = '-1'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'


def load_llama_model():
    llama_model = Llama.build(
        ckpt_dir=g_llama_model_path,
        tokenizer_path=g_tokenizer_path,
        max_seq_len=g_max_seq_len,
        max_batch_size=g_max_batch_size,
    )
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

    sys_msg = "Be rigorous and honest on every sentence in responses. Always keep consistency and coherence in the entire dialog. Always explain why if you detect inconsistency or incoherence in the dialog. If you do not know the answer to a question, do not output false information."

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
            sys_msg=sys_msg,
            max_gen_len=g_max_gen_len,
            temperature=g_temperature,
            top_p=g_top_p,
        )
        response_content = response[0]['generation']['content']
        logging.debug('[llama_chat] Iter: %s, Elapse: %s' % (iter_idx, time.time() - start_t))

        print('[%s Llama] %s' % (iter_idx, response_content))
        print('------------------------------------------------------------\n')

        llama_msg = {'role': response[0]['generation']['role'],
                     'content': response_content}
        full_dialog.append(llama_msg)
        iter_idx += 1

        user_content = 'Your response is incorrect. Explicitly identify incorrect statements, and correct them.'
        user_msg = {'role': 'user', 'content': user_content}
        full_dialog.append(user_msg)

    out_str = '\n'.join(['%s:%s' % (msg['role'], msg['content']) for msg in full_dialog])
    with open('controversial_conversation_%s.txt' % datetime.strftime(datetime.now(), '%Y%m%d%H%M%S'), 'w+') as out_fd:
        out_fd.write(out_str)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # Force to use CPU
    set_env_vars()
    # Retrieve Llama model
    llama_model = load_llama_model()
    # Chat completion
    llama_chat(llama_model)


