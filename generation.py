# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict
# [LeSaRDe Change Begins]
import logging
# [LeSaRDe Change Ends]

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)


# [LeSaRDe Change Begins] GPU -> CPU
# from llama.model import ModelArgs, Transformer
# from llama.tokenizer import Tokenizer
sys.path.append(os.path.abspath('/home/fmeng/workspace/llm_test'))
from model import ModelArgs, Transformer
from tokenizer import Tokenizer
# [LeSaRDe Change Ends]

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        if not torch.distributed.is_initialized():
            # [LeSaRDe Change Begins] GPU -> CPU
            # torch.distributed.init_process_group("nccl")
            torch.distributed.init_process_group("gloo")
            # [LeSaRDe Change Ends]
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # [LeSaRDe Change Begins] GPU -> CPU
        # torch.cuda.set_device(local_rank)
        # [LeSaRDe Change Ends]

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        # [LeSaRDe Change Begins] GPU -> CPU
        # assert model_parallel_size == len(
        #     checkpoints
        # ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        # [LeSaRDe Change Ends]
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        # [LeSaRDe Change Begins] GPU -> CPU
        # torch.set_default_tensor_type(torch.cuda.HalfTensor)
        torch.set_default_tensor_type(torch.FloatTensor)
        # [LeSaRDe Change Ends]
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    # [LeSaRDe Change Begins] Track Transformer behaviors
    # ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
    ) -> Tuple[List[List[int]], Optional[List[List[float]]], List[List[float]]]:
    # [LeSaRDe Change Ends]
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        # [LeSaRDe Change Begins] GPU -> CPU
        # tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cpu")
        # [LeSaRDe Change Ends]
        for k, t in enumerate(prompt_tokens):
            # [LeSaRDe Change Begins] GPU -> CPU
            # tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cpu")
            # [LeSaRDe Change Ends]
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        # [LeSaRDe Change Begins] GPU -> CPU
        # eos_reached = torch.tensor([False] * bsz, device="cuda")
        eos_reached = torch.tensor([False] * bsz, device="cpu")
        # [LeSaRDe Change Ends]
        input_text_mask = tokens != pad_id
        # [LeSaRDe Change Begins] Track Transformer behaviors
        l_h_seq = []
        # [LeSaRDe Change Ends]
        if min_prompt_len == total_len:
            # [LeSaRDe Change Begins] Track Transformer behaviors
            logging.error('[Llama:generate] min_prompt_len = total_len. min_prompt_len = %s, total_len = %s'
                          % (min_prompt_len, total_len))
            # logits = self.model.forward(tokens, prev_pos)
            logits, l_h = self.model.forward(tokens, prev_pos)
            l_h_seq.append(l_h)
            # [LeSaRDe Change Ends]
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        for cur_pos in range(min_prompt_len, total_len):
            # [LeSaRDe Change Begins] Track Transformer behaviors
            # logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            logits, l_h = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            l_h_seq.append(l_h)
            # [LeSaRDe Change Ends]
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        # [LeSaRDe Change Begins] Track Transformer behaviors
        # return (out_tokens, out_logprobs if logprobs else None)
        return (out_tokens, out_logprobs if logprobs else None, l_h_seq)
        # [LeSaRDe Change Begins]

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    # [LeSaRDe Change Begins] Track Transformer behaviors
    # ) -> List[CompletionPrediction]:
    ) -> Tuple[List[CompletionPrediction], List[List[float]]]:
    # [LeSaRDe Change Begins]
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        # generation_tokens, generation_logprobs = self.generate(
        #     prompt_tokens=prompt_tokens,
        #     max_gen_len=max_gen_len,
        #     temperature=temperature,
        #     top_p=top_p,
        #     logprobs=logprobs,
        #     echo=echo,
        # )
        # if logprobs:
        #     return [
        #         {
        #             "generation": self.tokenizer.decode(t),
        #             "tokens": [self.tokenizer.decode(x) for x in t],
        #             "logprobs": logprobs_i,
        #         }
        #         for t, logprobs_i in zip(generation_tokens, generation_logprobs)
        #     ]
        # return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]
        # [LeSaRDe Change Begins] Track Transformer behaviors
        generation_tokens, generation_logprobs, l_h_seq = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        # [LeSaRDe Change Ends]
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ], l_h_seq
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens], l_h_seq

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    # [LeSaRDe Change Begins] Track Transformer behaviors
    # ) -> List[ChatPrediction]:
    ) -> Tuple[List[ChatPrediction], List[List[float]]]:
    # [LeSaRDe Change Ends]
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)
        # [LeSaRDe Change Begins] Track Transformer behaviors
        # generation_tokens, generation_logprobs = self.generate(
        #     prompt_tokens=prompt_tokens,
        #     max_gen_len=max_gen_len,
        #     temperature=temperature,
        #     top_p=top_p,
        #     logprobs=logprobs,
        # )
        # if logprobs:
        #     return [
        #         {
        #             "generation": {
        #                 "role": "assistant",
        #                 "content": self.tokenizer.decode(t)
        #                 if not unsafe
        #                 else UNSAFE_ERROR,
        #             },
        #             "tokens": [self.tokenizer.decode(x) for x in t],
        #             "logprobs": logprobs_i,
        #         }
        #         for t, logprobs_i, unsafe in zip(
        #             generation_tokens, generation_logprobs, unsafe_requests
        #         )
        #     ]
        # return [
        #     {
        #         "generation": {
        #             "role": "assistant",
        #             "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
        #         }
        #     }
        #     for t, unsafe in zip(generation_tokens, unsafe_requests)
        # ]
        generation_tokens, generation_logprobs, l_h_seq = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        # [LeSaRDe Change Ends]
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ], l_h_seq
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ], l_h_seq

    # [LeSaRDe Change Begins] Address the prompt length issue
    def simple_chat_completion(self,
                               dialog: Dialog,
                               sys_msg: str = None,
                               temperature: float = 0.6,
                               top_p: float = 0.9,
                               max_gen_len: Optional[int] = None,
                               logprobs: bool = False,
                               keep_init_prompt: bool = True
                               ) -> Tuple[List[ChatPrediction], str, List[List[float]]]:
        """
        Always keeps the initial prompt, and tries to keep follow-ups as much as possible.
        """
        import logging
        import numpy as np

        if len(dialog) <= 0:
            logging.error('[simple_chat_completion] dialog is empty.')
            return None, None, None

        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        params = self.model.params
        prompt_tokens = []

        init_msg = dialog[0]
        if init_msg['role'] != 'user':
            logging.error('[simple_chat_completion] The initial user prompt is missing.')
            return None, None, None

        start_t = time.time()
        if sys_msg is not None:
            init_msg_content = ('%s %s %s %s %s %s'
                                % (B_INST, B_SYS, sys_msg.strip(), E_SYS, init_msg['content'], E_INST))
        else:
            init_msg_content = ('%s %s %s' % (B_INST, init_msg['content'], E_INST))
        init_prompt_tokens = self.tokenizer.encode(init_msg_content, bos=True, eos=True)

        for i in range(len(dialog)-1, 0, -1):
            msg = dialog[i]
            role = msg['role'].strip()
            content = msg['content'].strip()
            if role == 'user':
                tokens = self.tokenizer.encode('%s %s %s' % (B_INST, content, E_INST), bos=True, eos=True)
            elif role == 'assistant':
                tokens = self.tokenizer.encode(content, bos=True, eos=True)
            else:
                print('[simple_chat_completion] Unsupported role in msg: %s' % msg)
                tokens = []
            if keep_init_prompt:
                total_len = len(prompt_tokens) + len(init_prompt_tokens) + len(tokens)
            else:
                total_len = len(prompt_tokens) + len(tokens)
            if total_len < params.max_seq_len:
                prompt_tokens = tokens + prompt_tokens
            else:
                break
        if keep_init_prompt:
            prompt_tokens = init_prompt_tokens + prompt_tokens
        logging.debug('[simple_chat_completion] Prompt: [Elapse: %s, Len: %s]\n'
                      % (np.round(time.time() - start_t, decimals=4),
                         len(prompt_tokens)))

        generation_tokens, generation_logprobs, l_h_seq = self.generate(
            prompt_tokens=[prompt_tokens],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )

        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(
                    generation_tokens, generation_logprobs
                )
            ], self.tokenizer.decode(prompt_tokens), l_h_seq
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t),
                }
            }
            for t in generation_tokens
        ], self.tokenizer.decode(prompt_tokens), l_h_seq
    # [LeSaRDe Change Ends]

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
