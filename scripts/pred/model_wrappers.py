# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from typing import Dict, List, Optional

import requests
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("WARNING: vllm not found. LLM and SamplingParams undefined")

try:
    from lmcache_vllm.vllm import LLM, SamplingParams
except ImportError:
    print("WARNING: lmcache_vllm not found. LLM and SamplingParams undefined")

import torch


class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True
        )

        if "Yarn-Llama" in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {
                "attn_implementation": os.environ.get(
                    "ATTN_IMPLEMENTATION", "flash_attention_2"
                )
            }

        if os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2") in [
            "hip_attention",
            "flash_attention_2",
        ]:
            from hip_research.main.long_eval_decode_test import (Config,
                                                                 init_model)

            print(f"loading recompute model")

            recompute_n = 1024
            config = Config(
                model=name_or_path,
                recompute_n=recompute_n,
                long_ce_block_size=4096,
                long_ppl_alpha=2.0,
                long_ppl_beta=-2.0,
            )

            self.pipeline = None

            model, _ = init_model(config)
            self.model = model.cuda()
        elif "minference" in os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"):

            def init_model():
                device = "cuda:0"

                import transformers
                from minference import MInference

                attn_implementation = os.environ.get(
                    "ATTN_IMPLEMENTATION", "minference"
                )
                postfix = os.environ.get("USE_ATTN_POSTFIX", "none")
                if "delta" in postfix:
                    attn_implementation += "-delta"

                model_config = transformers.AutoConfig.from_pretrained(
                    name_or_path,
                    torch_dtype=torch.bfloat16,
                )

                model = transformers.AutoModelForCausalLM.from_pretrained(
                    name_or_path,
                    config=model_config,
                    torch_dtype=torch.bfloat16,
                )

                minference_patch = MInference(attn_implementation, name_or_path)
                model = minference_patch(model)

                return model, None

            model, _ = init_model()
            self.model = model.cuda()
            self.pipeline = None
        elif "cacheblend" in os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"):

            # Standard
            from dataclasses import asdict

            # Third Party
            from transformers import AutoTokenizer
            from vllm.config import KVTransferConfig
            from vllm.engine.arg_utils import EngineArgs

            def setup_environment_variables(use_disk: bool = False, blend_special_str: str = "# #"):
                # LMCache-related environment variables

                # LMCache is set to use 256 tokens per chunk
                os.environ["LMCACHE_CHUNK_SIZE"] = "256"

                # Blending related config
                os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
                os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
                os.environ["LMCACHE_USE_LAYERWISE"] = "True"

                if use_disk:
                    # Disable local CPU backend in LMCache
                    os.environ["LMCACHE_LOCAL_CPU"] = "False"

                    # Set the maximum size of the local CPU buffer size to 5GB
                    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

                    # Enable local disk backend in LMCache
                    os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"

                    # Set the maximum size of the local disk size to 10GB
                    os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "10"
                else:
                    # Enable local CPU backend in LMCache
                    os.environ["LMCACHE_LOCAL_CPU"] = "True"

                    # Set the maximum size of the local CPU size to 5GB
                    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"


            lmcache_connector = "LMCacheConnectorV1"
            ktc = KVTransferConfig(
                kv_connector=lmcache_connector,
                kv_role="kv_both",
            )

            llm_args = EngineArgs(
                model=name_or_path,
                kv_transfer_config=ktc,
                max_model_len=131072,
                gpu_memory_utilization=0.8,
                enable_prefix_caching=False,
                # swap_space=32,
                # cpu_offload_gb=32,
                tensor_parallel_size=2,
            )

            setup_environment_variables()
            self.model = LLM(**asdict(llm_args))
            self.pipeline = None

        elif "ape" in os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"):
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import numpy as np
            from ape import enable_attention_prefill_prefix, enable_attention_prefill_context, enable_attention_prefill_query

            self.model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda().eval()
            self.pipeline = None

            def build_prefix(model_name, prompt):
                if "llama" in model_name:
                    prompt = f"<|begin_of_text|>\n<|start_header_id|>user<|end_header_id|>\n{prompt}"
                elif "mistral" in model_name:
                    prompt = f"<s>[INST]{prompt}"
                elif "gemma" in model_name:
                    prompt = f"<bos><start_of_turn>user\n{prompt}"
                return prompt

            def build_suffix(model_name, prompt):
                if "llama" in model_name:
                    prompt = f"{prompt}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
                elif "mistral" in model_name:
                    prompt = f"{prompt}[/INST]"
                elif "gemma" in model_name:
                    prompt = f"{prompt}<end_of_turn>\n<start_of_turn>model\n"   
                return prompt

            def generate(prefix, contexts, query, model, tokenizer, temperature=0.9, scale=0.9):
                tokenizer.pad_token_id = self.tokenizer.eos_token_id
                # prefix = build_prefix(name_or_path, prefix)
                # query = build_suffix(name_or_path, query)

                with torch.no_grad():
                    prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
                    query_input_ids = tokenizer(query, truncation=False, return_tensors="pt").input_ids
                    len_prefix = prefix_input_ids.shape[1]
                    len_query = query_input_ids.shape[1]
                    context_input_ids = tokenizer(
                        contexts,
                        return_tensors='pt',
                        truncation=True,
                        max_length=131072,
                        padding=True,
                        add_special_tokens=False
                    ).input_ids

                    context_mask = (context_input_ids != tokenizer.pad_token_id).reshape(-1)
                    
                    enable_attention_prefill_prefix(name_or_path, model)
                    past_key_values = None
                    outputs = model(
                        prefix_input_ids.to(model.device),
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                    past_key_values = []
                    for past_key_value in outputs.past_key_values:
                        bsz, _ = context_input_ids.shape
                        past_key = past_key_value[0].repeat(bsz, 1, 1, 1)
                        past_value = past_key_value[1].repeat(bsz, 1, 1, 1)
                        past_position = past_key_value[2]
                        past_key_values.append([past_key, past_value, past_position])

                    enable_attention_prefill_context(name_or_path, model)
                    outputs = model(
                        context_input_ids.to(model.device),
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = []

                    for past_key_value in outputs.past_key_values:
                        bsz, num_heads, seq_len, _ = past_key_value[0].size()
                        past_key = torch.cat([past_key_value[0][:1, :, :len_prefix, :], 
                                                past_key_value[0][:, :, len_prefix:, :].transpose(1, 2).flatten(0, 1)[context_mask].unsqueeze(0).transpose(1, 2)], dim=2)
                        past_value = torch.cat([past_key_value[1][:1, :, :len_prefix, :], 
                                                past_key_value[1][:, :, len_prefix:, :].transpose(1, 2).flatten(0, 1)[context_mask].unsqueeze(0).transpose(1, 2)], dim=2)  
                        past_position = torch.cat([past_key_value[2][:, :len_prefix],
                                                    past_key_value[2][:, len_prefix:].repeat(bsz, 1).flatten()[context_mask].unsqueeze(0)], dim=1)
                        past_key_values.append([past_key, past_value, past_position, len(contexts)])
                    
                    context_input_ids = context_input_ids.flatten()[context_mask].unsqueeze(0)
                    input_ids = torch.cat([prefix_input_ids, context_input_ids, query_input_ids], dim=-1)
                    context_length = input_ids.shape[-1]

                    enable_attention_prefill_query(name_or_path, model, temperature, scale)
                    generation_kwargs = {}
                    generation_kwargs["cache_implementation"] = None
                    output = model.generate(
                        input_ids=input_ids.to(model.device),
                        max_new_tokens=128,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        past_key_values=past_key_values,
                        **generation_kwargs
                    )[0]
                    pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
                    print(f"APE prediction: {pred}")
                    return pred

            self.generate = generate

        # ORIGINAL ===============================================
        # try:
        #     self.pipeline = pipeline(
        #         "text-generation",
        #         model=name_or_path,
        #         tokenizer=self.tokenizer,
        #         trust_remote_code=True,
        #         device_map="auto",
        #         torch_dtype=torch.bfloat16,
        #         model_kwargs=model_kwargs,
        #     )
        # except:
        #     self.pipeline = None
        #     self.model = AutoModelForCausalLM.from_pretrained(
        #         name_or_path,
        #         trust_remote_code=True,
        #         device_map="auto",
        #         torch_dtype=torch.bfloat16,
        #     )

        print(f"{generation_kwargs=}")
        # {
        #  'do_sample': False,
        #  'repetition_penalty': 1,
        #  'temperature': 0.0,
        #  'top_k': 32,
        #  'top_p': 1.0,
        #  'stop': [],
        #  'max_new_tokens': 128
        # }
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        if self.pipeline is None:
            if "cacheblend" in os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"):

                prompt = prompts[0]
                seq_len = int(os.environ.get("MAX_SEQ_LENGTH", "131072"))
                n_chunks = seq_len // 2048
                print(f"\n\n{seq_len=} {n_chunks=}\n\n")
                chunk_size = len(prompt) // n_chunks
                chunks = [self.tokenizer.encode(prompt[i:i+chunk_size])[1:] for i in range(0, len(prompt), chunk_size)]
                blend_special_str = self.tokenizer.encode(os.getenv("LMCACHE_BLEND_SPECIAL_STR"))[1:]

                chunks[1] = chunks[0][16:] + chunks[1]
                chunks[0] = chunks[0][:16]
                if len(chunks[-1]) < 2048:
                    chunks[-2] += chunks[-1]
                    chunks = chunks[:-1]

                # do this to induce individial caching of blocks
                sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
                for i, c in enumerate(chunks[:-1]):
                    p = c + blend_special_str
                    outputs_cached = self.model.generate(prompt_token_ids=p, sampling_params=sampling_params)

                # prompt1 = []
                # for i, c in enumerate(chunks[:-1]):
                #     if i < len(chunks) - 1:
                #         prompt1 += c + blend_special_str
                #     else:
                #         prompt1 += c

                prompt2 = []
                for i, c in enumerate(chunks):
                    if i < len(chunks) - 1:
                        prompt2 += c + blend_special_str
                    else:
                        prompt2 += c

                # print(f"{prompt=}")

                sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=128)
                outputs = self.model.generate(prompt_token_ids=prompt2, sampling_params=sampling_params)
                print("-" * 50)
                for output in outputs:
                    generated_text = output.outputs[0].text
                    print(f"Generated text: {generated_text!r}")

                generated_texts = [outputs[-1].outputs[0].text]

            elif "ape" in os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"):
                prompt = prompts[0]
                seq_len = int(os.environ.get("MAX_SEQ_LENGTH", "131072"))
                n_chunks = seq_len // 2048
                chunk_size = len(prompt) // n_chunks
                chunks = [prompt[i:i + chunk_size] for i in range(0, len(prompt), chunk_size)]

                prefix, contexts, query = chunks[0][:512], [chunks[0][512:]] + chunks[:len(chunks) - 1], chunks[-1]

                if len(query) < chunk_size // 4:
                    query += contexts[-1][-chunk_size // 4:]
                    contexts[-1] = contexts[-1][:-chunk_size // 4]

                # print(f"{len(contexts)=}")
                # print(f"\n{contexts}\n")

                # print(f"{prefix=} \n\n {query=}")
                # for c in contexts:
                #     print(f"\n\n{c}")

                # prefix = ""
                # contexts = [
                #     "My friends and I love going on road trips and exploring new places. However, we also enjoy hiking and camping together in nature.",
                #     "We often spend time playing board games and card games as a group. But we also like solving escape rooms and participating in trivia nights.",
                #     "Many of my friends enjoy listening to live music and attending concerts. We also love discovering new artists and sharing playlists with each other.",
                #     "We like trying out different coffee shops and bakeries. However, we also enjoy experimenting with baking and making homemade desserts.",
                #     "My friends and I love learning new skills, like photography and painting. We also enjoy visiting art workshops and DIY craft events."
                # ]
                # query = "Question: what are ten ideas for a social with a large groups of friends in New York City.\nAnswer:"

                generated_texts = self.generate(prefix, contexts, query, self.model, self.tokenizer)
                generated_texts = [generated_texts]

            else:
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
                    self.model.device
                )
                generated_ids = self.model.generate(**inputs, **self.generation_kwargs)
                generated_texts = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
        else:
            output = self.pipeline(
                text_inputs=prompts,
                **self.generation_kwargs,
            )
            assert len(output) == len(prompts)
            # output in the form of a list of list of dictionaries
            # outer list len = batch size
            # inner list len = 1
            generated_texts = [llm_result[0]["generated_text"] for llm_result in output]

        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            # This is a workaround for the llama3 tokenizer not being able to reproduce the same prompt after tokenization
            # see Issue https://github.com/NVIDIA/RULER/issues/54 for explaination
            if self.pipeline is None:
                tokenized_prompt = self.tokenizer(
                    prompt, return_tensors="pt", padding=True
                )
                prompt = self.tokenizer.decode(
                    tokenized_prompt.input_ids[0], skip_special_tokens=True
                )
            if text.startswith(prompt):
                text = text[len(prompt) :]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({"text": [text]})

        return results


class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model = MambaLMHeadModel.from_pretrained(
            name_or_path, device=self.device, dtype=torch.bfloat16
        )
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")
        self.max_genlen = self.generation_kwargs.pop("max_new_tokens")
        self.minp = 0.0

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {"text": [self.tokenizer.decode(out.sequences[0][input_ids.shape[1] :])]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        # FIXME: naive implementation
        return [self.__call__(prompt, **kwargs) for prompt in prompts]
