import os.path
import signal
import sys
from datetime import datetime
import time
import numpy as np
# from llama import Llama
from llama_cpp import Llama

g_model_path = '/home/fmeng/workspace/llama_models/llama-2-7b-chat/ggml-model-q5_k_m.gguf'
# g_model_path = '/home/fmeng/workspace/llama_models/llama-2-13b-chat/ggml-model-f16.gguf'
# g_model_path = '/home/fmeng/workspace/llama_models/llama-2-7b-chat-uncensored/ggml-model-q5_k_m.gguf'
g_tokenizer_path = '/home/fmeng/workspace/llama_models/tokenizer.model'
# g_tokenizer_path = '/home/fmeng/workspace/llama_models/llama-2-7b-chat-uncensored/tokenizer.model'

# g_sys_msg = "Your responses are always relevant to the first user prompt. Be rigorous and honest on every sentence in responses. Always keep consistency and coherence in the entire dialog. Always explain why if you detect inconsistency or incoherence in the dialog. If you do not know the answer to a question, do not output false information or irrelevant information."
g_sys_msg = None

# g_contro_prompt = """Your last response includes incorrect information. Explicitly identify incorrect statements, and correct them."""
g_contro_prompt = """More."""


def llama_chat():
    def __sigint_handler(sig, frame):
        print('''\nQuit.''')
        print('------------------------------------------------------------\n')
        sys.exit(0)

    signal.signal(signal.SIGINT, __sigint_handler)

    if os.path.exists('TERM'):
        os.remove('TERM')

    context_len = 2048
    n_gpu_layers = -1
    max_tokens = None
    temperature = 1.0
    top_p = 0.9
    presence_penalty = None
    frequency_penalty = None
    repeat_penalty = None
    verbose = False

    llm = Llama(
        g_model_path,
        n_ctx=context_len,
        n_gpu_layers=n_gpu_layers,
        n_threads=os.cpu_count(),
        n_threads_batch=os.cpu_count(),
        seed=-1,
        verbose=verbose)

    print('You will start this conversation.\n')
    start_str = input('[%s You]: ' % 0)
    print('------------------------------------------------------------\n')

    init_msg = {'role': 'user', 'content': start_str}

    if g_sys_msg is not None:
        sys_msg = {
            "role": "system",
            "content": g_sys_msg
        }
        l_full_history = [sys_msg, init_msg]
    else:
        l_full_history = [init_msg]

    contro_msg = {'role': 'user', 'content': g_contro_prompt}

    iter_idx = 0
    while True:
        if os.path.exists('TERM'):
            break

        # l_full_history.append(your_msg)
        start_t = time.time()
        if g_sys_msg is not None:
            if len(l_full_history) >= 6:
                chat = [sys_msg, init_msg,
                        l_full_history[2], # first Llama response
                        l_full_history[-3], # last contro_msg
                        l_full_history[-2], # last Llama response to last contro_msg
                        l_full_history[-1]  # new contro_msg
                        ]
            else:
                chat = l_full_history
        else:
            if len(l_full_history) >= 5:
                chat = [init_msg,
                        l_full_history[1],  # first Llama response
                        l_full_history[-3],  # last contro_msg
                        l_full_history[-2],  # last Llama response to last contro_msg
                        l_full_history[-1]   # new contro_msg
                        ]
            else:
                chat = l_full_history

        response = llm.create_chat_completion(
            chat,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            # presence_penalty=presence_penalty,
            # frequency_penalty=frequency_penalty,
            # repeat_penalty=repeat_penalty
        )

        llama_msg = response['choices'][0]['message']
        print('[%s Llama] (Elapse: %s) %s' % (iter_idx, np.round(time.time() - start_t, decimals=2),
                                              llama_msg['content']))
        print('------------------------------------------------------------\n')
        l_full_history.append(llama_msg)

        iter_idx += 1
        print('[%s You] %s' % (iter_idx, contro_msg['content']))
        print('------------------------------------------------------------\n')
        l_full_history.append(contro_msg)


    if g_sys_msg is not None:
        out_str = '\n\n'.join(['%s:%s:%s' %
                               (int((idx-1)/2) if idx % 2 != 0 and idx > 0 else int(idx/2),
                                msg['role'],
                                msg['content']) for idx, msg in enumerate(l_full_history[1:])])
        out_str = '%s\n\n%s' % (l_full_history[0], out_str)
    else:
        out_str = '\n\n'.join(['%s:%s:%s' %
                               (int((idx - 1) / 2) if idx % 2 != 0 and idx > 0 else int(idx / 2),
                                msg['role'],
                                msg['content']) for idx, msg in enumerate(l_full_history)])
    with open('quant_llama_contro_test_%s.txt' % datetime.strftime(datetime.now(), '%Y%m%d%H%M%S'), 'w+') as out_fd:
        out_fd.write(out_str)
    print('''\nAll done.''')
    print('------------------------------------------------------------\n')


if __name__ == '__main__':
    llama_chat()