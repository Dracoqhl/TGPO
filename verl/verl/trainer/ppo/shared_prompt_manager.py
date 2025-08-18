# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Utilities to manage shared textual prompts for language-gradient training.

The :class:`SharedPromptManager` maintains two pieces of text information that
are prepended to every chat prompt during training:

``global_knowledge``
    A persistent description summarising what the model has learnt so far.
``current_gradient``
    A short natural language feedback for the most recent update step.

This manager exposes three main interfaces:

``build_prefix``
    Concatenate the two sections into a single system message string.
``augment_prompt``
    Insert the system message to a chat ``messages`` list and tokenize it to
    produce model inputs compatible with the rest of the training pipeline.
``update``
    Call an external analyser to refresh ``global_knowledge`` and
    ``current_gradient`` from the latest prompts, responses and rewards.

The external analyser is provided by the caller and is expected to be a
callable returning a dictionary with ``global_knowledge`` and
``current_gradient`` fields.  The actual implementation (e.g. calling another
LLM) is intentionally kept outside of this module.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, List, Dict, Any, Optional

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask


logger = logging.getLogger(__name__)


class SharedPromptManager:
    """Manage a shared textual prefix for PPO style training.

    Parameters
    ----------
    tokenizer:
        Tokenizer used to convert messages to model inputs.
    analysis_fn:
        Callable invoked by :meth:`update`.  It should accept keyword
        arguments ``prompts``, ``responses`` and ``rewards`` and return a
        mapping containing ``global_knowledge`` and ``current_gradient``.
    max_prompt_length:
        Optional maximum length for tokenised prompts.  Defaults to
        ``tokenizer.model_max_length``.
    truncation:
        Truncation strategy passed to :func:`verl.utils.torch_functional.postprocess_data`.
    """

    def __init__(
        self,
        tokenizer,
        analysis_fn: Optional[Callable[..., Dict[str, str]]] = None,
        *,
        max_prompt_length: Optional[int] = None,
        truncation: str = "left",
    ) -> None:
        self.tokenizer = tokenizer
        self.analysis_fn = analysis_fn
        self.max_prompt_length = max_prompt_length or getattr(tokenizer, "model_max_length", None)
        self.truncation = truncation

        # The shared textual information
        self.global_knowledge: str = ""
        self.current_gradient: str = ""

    # ------------------------------------------------------------------
    # Building and inserting prefix
    # ------------------------------------------------------------------
    def build_prefix(self) -> str:
        """Return the concatenated system prompt.

        Returns
        -------
        str
            A single string that joins ``global_knowledge`` and
            ``current_gradient``.  An empty string is returned if both parts
            are empty.
        """

        sections: List[str] = []
        if self.global_knowledge:
            sections.append(f"[Global Knowledge]\n{self.global_knowledge}")
        if self.current_gradient:
            sections.append(f"[Current Gradient]\n{self.current_gradient}")
        return "\n\n".join(sections)

    def augment_prompt(self, messages: Iterable[Dict[str, str]]) -> Dict[str, Any]:
        """Insert shared prefix into ``messages`` and tokenize.

        Parameters
        ----------
        messages:
            Chat messages as a list of ``{"role": ..., "content": ...}``.

        Returns
        -------
        dict
            Dictionary containing ``input_ids``, ``attention_mask``,
            ``position_ids`` and ``raw_prompt_ids`` compatible with the
            training pipeline.  The possibly augmented messages are also
            returned under ``raw_prompt`` for downstream use.
        """

        # Ensure ``messages`` is a list for further processing.
        msgs: List[Dict[str, str]] = list(messages)
        prefix = self.build_prefix()
        if prefix:
            msgs = [{"role": "system", "content": prefix}] + msgs

        raw_prompt = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )

        position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if self.max_prompt_length is not None and len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "raw_prompt": msgs,
        }

    # ------------------------------------------------------------------
    # Updating shared information
    # ------------------------------------------------------------------
    def update(
        self,
        prompts: Iterable[str],
        responses: Iterable[str],
        rewards: Iterable[float],
    ) -> None:
        """Update shared prefix via the external analysis function.

        Parameters
        ----------
        prompts, responses, rewards:
            Data from the latest rollout step passed to ``analysis_fn``.
        """

        if self.analysis_fn is None:
            return

        try:
            result = self.analysis_fn(
                prompts=list(prompts),
                responses=list(responses),
                rewards=list(rewards),
                global_knowledge=self.global_knowledge,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("SharedPromptManager.update failed")
            return

        if not isinstance(result, dict):
            logger.warning("analysis_fn should return a dict, got %r", type(result))
            return

        self.global_knowledge = result.get("global_knowledge", self.global_knowledge) or ""
        self.current_gradient = result.get("current_gradient", self.current_gradient) or ""
