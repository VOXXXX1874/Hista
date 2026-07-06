from trl.generation.vllm_generation import VLLMGeneration, empty_cache, extract_logprobs

import json
import logging
from contextlib import nullcontext

import torch
from accelerate.utils import broadcast_object_list, gather_object
from packaging.version import Version

from trl.data_utils import apply_chat_template, is_conversational, prepare_multimodal_messages_vllm
from trl.extras.profiling import ProfilingContext
from trl.import_utils import is_vllm_available


if is_vllm_available():
    import vllm
    from vllm import SamplingParams

    if Version(vllm.__version__) <= Version("0.10.2"):
        from vllm.sampling_params import GuidedDecodingParams
    else:
        from vllm.sampling_params import StructuredOutputsParams


logger = logging.getLogger(__name__)

# VLLM Generation with sleep bug fixed
class VLLMGenerationPatched(VLLMGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.enable_sleep_mode:
            self._llm_weights_sleeping = True
        else:
            self._llm_weights_sleeping = False

    def sync_weights(self):
        super().sync_weights()
        self._llm_weights_sleeping = False

    def generate(self, prompts: list, num_generations: int, profiler: ProfilingContext | None = None) -> tuple:
        """Generate completions using vLLM.

        Args:
            prompts: List of prompts (strings or chat conversations)
            num_generations: Number of generations per prompt
            profiler: Optional profiler for performance tracking

        Returns:
            Tuple of (prompt_ids, completion_ids, logprobs, logprob_token_ids, extra_fields).

            - `prompt_ids`: `list[list[int]]` of shape `(batch_size, prompt_len)`.
            - `completion_ids`: `list[list[int]]` of shape `(batch_size, completion_len)`.
            - `logprobs`: `list[list[list[float | None]]]` of shape `(batch_size, completion_len, num_logprobs)`.
            - `logprob_token_ids`: `list[list[list[int]]]` of shape `(batch_size, completion_len, num_logprobs)`.
            - `extra_fields`: `dict` of additional per-completion fields from a custom `rollout_func`.

            `num_logprobs` is 1 when `logprobs=0`, or up to N+1 when `logprobs=N` (the sampled token is always included
            and may fall outside the top-N).
        """
        profiler = profiler or nullcontext()
        accelerator = self.accelerator
        rollout_func = self.rollout_func
        temperature = self.temperature
        top_p = self.top_p
        top_k = self.top_k
        min_p = self.min_p
        repetition_penalty = self.repetition_penalty
        max_completion_length = self.max_completion_length
        processing_class = self.processing_class
        chat_template_kwargs = self.chat_template_kwargs
        tools = self.tools
        chat_template = self.chat_template

        # Wake up colocated vLLM weights if needed (idempotent if already awake from sync_weights)
        if self.mode == "colocate" and self.enable_sleep_mode and self._llm_weights_sleeping:
            empty_cache()  # required to avoid OOM in some cases
            self.llm.wake_up(tags=["weights"])
            # Work around for https://github.com/vllm-project/vllm/issues/29341
            try:
                self.llm.collective_rpc("reload_weights")
            except NotImplementedError:
                # Non-CUDA vLLM backends (e.g., vllm-ascend's NPUWorkerV1), don't implement reload_weights
                pass

        if is_conversational({"prompt": prompts[0]}):
            prompts = [prepare_multimodal_messages_vllm(prompt) for prompt in prompts]

        # In vLLM, tool call arguments must be JSON strings. See https://github.com/vllm-project/vllm/pull/28820
        for prompt in prompts:  # iterate over each conversation
            if is_conversational({"prompt": prompt}):
                for message in prompt:  # iterate over each message
                    if "tool_calls" in message:  # check if message has tool calls
                        for call in message["tool_calls"]:
                            args_value = call["function"]["arguments"]
                            if isinstance(args_value, dict):  # only convert dict → JSON string
                                call["function"]["arguments"] = json.dumps(args_value)

        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        if self.mode == "server":
            all_prompts = gather_object(prompts)

            if accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                ordered_set_of_prompts = all_prompts[::num_generations]

                sampling_params = {
                    "n": num_generations,
                    "repetition_penalty": repetition_penalty,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": 0.0 if min_p is None else min_p,
                    "max_tokens": max_completion_length,
                    "logprobs": self.logprobs,
                    "structured_outputs_regex": self.structured_outputs_regex,
                    "generation_kwargs": self.generation_kwargs,
                }
                with profiler:  # TODO: profiling_context(trainer, "vLLM.generate"):
                    if rollout_func is not None:
                        # Pass all prompts (with duplicates) to rollout_func for consistency with colocate mode
                        rollout_prompts = all_prompts
                        if rollout_prompts and is_conversational({"prompt": rollout_prompts[0]}):
                            rollout_prompts = [
                                apply_chat_template({"prompt": p}, processing_class, **chat_template_kwargs)["prompt"]
                                for p in rollout_prompts
                            ]
                        output = rollout_func(rollout_prompts)
                    else:
                        if is_conversational({"prompt": ordered_set_of_prompts[0]}):
                            output = self.vllm_client.chat(
                                messages=ordered_set_of_prompts,
                                **sampling_params,
                                chat_template_kwargs=chat_template_kwargs,
                                tools=tools,
                                chat_template=chat_template,
                            )
                        else:
                            output = self.vllm_client.generate(prompts=ordered_set_of_prompts, **sampling_params)
                    # Extract required fields and collect any extra fields for reward functions
                    required_keys = {"prompt_ids", "completion_ids", "logprobs", "logprob_token_ids"}
                    extra_fields = {k: v for k, v in output.items() if k not in required_keys}
                    payload = (
                        output["prompt_ids"],
                        output["completion_ids"],
                        output["logprobs"],
                        output.get("logprob_token_ids"),
                        extra_fields,
                    )
            else:
                payload = None

            # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
            obj_list = [payload]
            broadcast_object_list(obj_list, from_process=0)
            all_prompt_ids, all_completion_ids, all_logprobs, all_logprob_token_ids, all_extra_fields = obj_list[0]

            # When using rollout_func, it handles its own generation logic and returns one result per prompt.
            # When NOT using rollout_func, vllm_client.generate(n=num_generations) returns num_generations
            # completions per prompt, so we need to duplicate prompt_ids to match.
            if self.rollout_func is None:
                # At this point, we only get 1 copy of each prompt, so we need to repeat them num_generations times
                all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(num_generations)]

            process_slice = slice(
                accelerator.process_index * len(prompts),
                (accelerator.process_index + 1) * len(prompts),
            )
            prompt_ids = all_prompt_ids[process_slice]
            completion_ids = all_completion_ids[process_slice]
            logprobs = all_logprobs[process_slice] if all_logprobs is not None else None
            logprob_token_ids = all_logprob_token_ids[process_slice] if all_logprob_token_ids is not None else None

            # Slice extra fields dict-of-lists per process (extra fields are per-completion, like completion_ids)
            extra_fields = {}
            for key, values in all_extra_fields.items():
                if isinstance(values, list):
                    extra_fields[key] = values[process_slice]
                else:
                    extra_fields[key] = values

        # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
        elif self.mode == "colocate":
            if rollout_func is not None:
                rollout_prompts = prompts
                if rollout_prompts and is_conversational({"prompt": rollout_prompts[0]}):
                    rollout_prompts = [
                        apply_chat_template({"prompt": prompt}, processing_class, **chat_template_kwargs)["prompt"]
                        for prompt in rollout_prompts
                    ]
                output = rollout_func(rollout_prompts)
                required_keys = {"prompt_ids", "completion_ids", "logprobs", "logprob_token_ids"}
                extra_fields = {k: v for k, v in output.items() if k not in required_keys}
                prompt_ids = output["prompt_ids"]
                completion_ids = output["completion_ids"]
                logprobs = output["logprobs"]
                logprob_token_ids = output.get("logprob_token_ids")
            else:
                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": repetition_penalty,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": 0.0 if min_p is None else min_p,
                    "max_tokens": max_completion_length,
                    "logprobs": self.logprobs,
                }
                generation_kwargs.update(self.generation_kwargs)

                if Version(vllm.__version__) <= Version("0.10.2"):
                    structured_outputs_key = "guided_decoding"
                    if self.structured_outputs_regex is not None:
                        if generation_kwargs.get("guided_decoding") is not None:
                            logger.warning(
                                "Both `structured_outputs_regex` and `generation_kwargs['guided_decoding']` are set; "
                                "`structured_outputs_regex` takes precedence."
                            )
                        structured_outputs = GuidedDecodingParams(regex=self.structured_outputs_regex)
                    else:
                        structured_outputs = generation_kwargs.get("guided_decoding")
                else:
                    structured_outputs_key = "structured_outputs"
                    if self.structured_outputs_regex is not None:
                        if generation_kwargs.get("structured_outputs") is not None:
                            logger.warning(
                                "Both `structured_outputs_regex` and `generation_kwargs['structured_outputs']` are "
                                "set; `structured_outputs_regex` takes precedence."
                            )
                        structured_outputs = StructuredOutputsParams(regex=self.structured_outputs_regex)
                    elif isinstance(generation_kwargs.get("structured_outputs"), dict):
                        structured_outputs_dict = generation_kwargs.get("structured_outputs")
                        structured_outputs = StructuredOutputsParams(**structured_outputs_dict)
                    else:
                        structured_outputs = generation_kwargs.get("structured_outputs")

                generation_kwargs[structured_outputs_key] = structured_outputs
                sampling_params = SamplingParams(**generation_kwargs)

                if self.tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts)
                    gathered_prompts = [None for _ in range(self.tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.tp_group)
                    all_prompts = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts = prompts

                if self.enable_sleep_mode:
                    self.llm.wake_up(tags=["kv_cache"])

                with profiler:  # TODO: profiling_context(trainer, "vLLM.generate"):
                    if is_conversational({"prompt": prompts[0]}):
                        all_outputs = self.llm.chat(
                            all_prompts,
                            sampling_params=sampling_params,
                            use_tqdm=False,
                            chat_template_kwargs=chat_template_kwargs,
                            tools=tools,
                            chat_template=chat_template,
                        )
                    else:
                        all_outputs = self.llm.generate(all_prompts, sampling_params=sampling_params, use_tqdm=False)

                all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
                all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
                all_logprobs, all_logprob_token_ids = extract_logprobs(all_outputs)

                if self.tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    prompt_ids = all_prompt_ids[tp_slice]
                    completion_ids = all_completion_ids[tp_slice]
                    logprobs = all_logprobs[tp_slice] if all_logprobs is not None else None
                    logprob_token_ids = all_logprob_token_ids[tp_slice] if all_logprob_token_ids is not None else None
                else:
                    prompt_ids = all_prompt_ids
                    completion_ids = all_completion_ids
                    logprobs = all_logprobs
                    logprob_token_ids = all_logprob_token_ids

                extra_fields = {}  # No extra fields for colocate mode

                if self.enable_sleep_mode:
                    self.llm.sleep(level=2)
                    self._llm_weights_sleeping = True

        return prompt_ids, completion_ids, logprobs, logprob_token_ids, extra_fields
