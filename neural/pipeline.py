from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import os

__all__ = ["Pipeline", "LMPipeline"]

# ---------------------------------------------------------------------------
# Helper utils
# ---------------------------------------------------------------------------

def _infer_device_and_dtype(requested_device=None, requested_dtype=None):
    """Return a sensible `(device, dtype)` pair when not fully specified."""
    if requested_device is None:
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_dtype is None:
        requested_dtype = torch.float16 if requested_device.startswith("cuda") else torch.float32
    return requested_device, requested_dtype


# ---------------------------------------------------------------------------
# Base pipeline – minimal signatures (no *args / **kwargs)
# ---------------------------------------------------------------------------

class Pipeline(ABC):
    """Abstract base pipeline.

    Subclasses must implement the hooks below. The base class deliberately
    avoids variadic parameters so implementers have full freedom to define
    their own concrete signatures.
    """

    def __init__(self, model_or_name):
        self.model_or_name = model_or_name
        self._setup_model()

    # ------------------------------------------------------------------
    # Abstract hooks – simple signatures only
    # ------------------------------------------------------------------

    @abstractmethod
    def _setup_model(self):
        pass

    @abstractmethod
    def load(self, raw_input):
        pass

    @abstractmethod
    def dump(self, model_output):
        pass

    @abstractmethod
    def generate(self, prompt):
        pass

    @abstractmethod
    def intervenable_generate(
        self,
        intervenable_model,
        base,
        sources,
        map,  # noqa: A002 – intentional name
        feature_indices,
    ):
        pass
        pass


# ---------------------------------------------------------------------------
# Language‑model pipeline (typed; unchanged implementation)
# ---------------------------------------------------------------------------

class LMPipeline(Pipeline):
    """Pipeline for autoregressive HuggingFace causal‑LMs."""

    def __init__(
        self,
        model_or_name: str | Any,
        *,
        max_new_tokens: int = 3,
        max_length: int | None = None,
        logit_labels: bool = False,
        position_ids: bool = False,
        use_chat_template: bool = False,
        **kwargs: Any,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.logit_labels = logit_labels
        self.position_ids = position_ids
        self.use_chat_template = use_chat_template
        # pass through kwargs to _setup_model via instance vars
        self._init_extra_kwargs = kwargs
        super().__init__(model_or_name)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        device, dtype = _infer_device_and_dtype(
            self._init_extra_kwargs.get("device"), self._init_extra_kwargs.get("dtype")
        )

        if isinstance(self.model_or_name, str):
            hf_token = self._init_extra_kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_or_name, token=hf_token)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_or_name, config=self._init_extra_kwargs.get("config"), token=hf_token
            ).to(device=device, dtype=dtype)
            if hasattr(self.model.config, "_attn_implementation") and "qwen" not in self.model.config.name_or_path.lower():
                self.model.config._attn_implementation = "eager"
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False
        else:
            self.model = self.model_or_name.to(device).to(dtype)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.name_or_path)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def load(
        self,
        input: Union[Dict, List[Dict], str, List[str]],
        *,
        max_length: int | None = None,
        padding_side: str | None = None,
        add_special_tokens: bool = True,
        use_chat_template: bool | None = None,
    ) -> Dict[str, torch.Tensor]:

        if use_chat_template is None:
            use_chat_template = self.use_chat_template

        if isinstance(input, str):
            input = [{"raw_input": input}]
        elif isinstance(input, list) and len(input) > 0 and isinstance(input[0], str):
            input = [{"raw_input": p} for p in input]

        if isinstance(input, Dict):
            assert "raw_input" in input, "Input dictionary must contain 'raw_input' key."
            raw_input = [input["raw_input"]]
        else:
            assert isinstance(input, list) or isinstance(input, tuple), "Input must be a dictionary or a list/tuple of dictionaries."
            assert all("raw_input" in item for item in input), "Each input dictionary must contain 'raw_input' key."
            raw_input = [item["raw_input"] for item in input]

        # Apply chat template if requested
        if use_chat_template:
            processed_input = []
            for text in raw_input:
                # Convert to messages format and apply chat template
                messages = [{"role": "user", "content": text}]
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                processed_input.append(formatted)
            raw_input = processed_input

        if max_length is None:
            max_length = self.max_length

        if padding_side is not None:
            prev_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = padding_side

        enc = self.tokenizer(
            raw_input,
            padding="max_length" if max_length else True,
            max_length=max_length,
            truncation=max_length is not None,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        if self.position_ids:
            enc["position_ids"] = self.model.prepare_inputs_for_generation(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
            )["position_ids"]
        for k, v in enc.items():
            if isinstance(v, torch.Tensor):
                enc[k] = v.to(self.model.device)
        
        if padding_side is not None:
            self.tokenizer.padding_side = prev_padding_side
        return enc

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def dump(
        self,
        model_output: torch.Tensor | list | tuple | Dict[str, Any],
        *,
        is_logits: bool = True,
    ) -> Union[str, List[str]]:
        if isinstance(model_output, dict):
            model_output = model_output.get("sequences", model_output.get("scores"))
            if isinstance(model_output, torch.Tensor):
                is_logits = model_output.dim() >= 3

        if isinstance(model_output, (list, tuple)):
            model_output = (
                model_output[0].unsqueeze(1) if len(model_output) == 1 else torch.stack(model_output, dim=1)
            )

        if isinstance(model_output, torch.Tensor):
            if model_output.dim() >= 3 and is_logits:
                token_ids = model_output.argmax(dim=-1)
            elif model_output.dim() == 2:
                token_ids = model_output
            elif model_output.dim() == 1:
                token_ids = model_output.unsqueeze(0)
            else:
                raise ValueError("Unexpected output shape for dump().")
        else:
            raise TypeError("model_output must be Tensor / list / tuple / dict")

        decoded = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return decoded[0] if len(decoded) == 1 else decoded

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, input: Union[Dict, List[Dict], str, List[str]], **gen_kwargs: Any) -> Dict[str, Any]:
        # Handle backward compatibility for raw strings
        inputs = self.load(input)
        defaults: dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            use_cache=False,
        )
        defaults.update(gen_kwargs)
        with torch.no_grad():
            out = self.model.generate(**inputs, **defaults)
        scores = [s.detach().cpu() for s in (out.scores or [])]
        seq = out.sequences[:, -self.max_new_tokens :].detach().cpu()
        del inputs, out
        torch.cuda.empty_cache()
        gc.collect()
        return {"scores": scores, "sequences": seq}

    # ------------------------------------------------------------------
    # Intervention generation
    # ------------------------------------------------------------------

    def intervenable_generate(
        self,
        intervenable_model: "IntervenableModel",  # type: ignore  # noqa: F821
        base: Any,
        sources: Any,
        map: Any,
        feature_indices: Any,
        *,
        output_scores: bool = False,
        **gen_kwargs: Any,
    ) -> torch.Tensor:
        defaults = dict(
            unit_locations=map,
            subspaces=feature_indices,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=output_scores,
            intervene_on_prompt=True,
            do_sample=False,
            use_cache=False,
        )
        defaults.update(gen_kwargs)
        with torch.no_grad():
            out = intervenable_model.generate(base, sources=sources, **defaults)
        return out[-1].scores if output_scores else out[-1].sequences[:, -self.max_new_tokens :]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_num_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)
