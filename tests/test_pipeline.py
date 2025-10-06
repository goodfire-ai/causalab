import types
import torch
import pytest

# ---------------------------------------------------------------------------
# Dummy HuggingFace‑like stubs
# ---------------------------------------------------------------------------

class DummyTokenizer:
    """Mimics the minimal HF tokenizer API used in `LMPipeline`."""

    pad_token = "<|endoftext|>"
    eos_token = pad_token

    def __init__(self):
        # simplistic vocab: each char → its ord() value
        self._vocab = {chr(i): i for i in range(97, 123)}  # a‑z
        self.pad_token_id = self.convert_tokens_to_ids(self.pad_token)
        self.padding_side = "right"

    # ------------------------------------------------------------------
    # HF‑style helpers required by LMPipeline
    # ------------------------------------------------------------------

    def encode(self, text):
        return [self.convert_tokens_to_ids(c) for c in text]

    def __call__(self, texts, *, padding, max_length, truncation, return_tensors, add_special_tokens):
        # Very naive: represent each string as its ord() ids, pad / truncate.
        batch = [self.encode(t) for t in texts]
        if max_length:
            batch = [seq[: max_length] + [self.pad_token_id] * (max_length - len(seq)) for seq in batch]
        # Build tensors
        input_ids = torch.tensor(batch, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(self, ids, skip_special_tokens=True):
        results = []
        for seq in ids:
            chars = [chr(int(i)) for i in seq.tolist() if i not in (self.pad_token_id, 0)]
            results.append("".join(chars))
        return results

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, 0)


class DummyGenerateOutput(types.SimpleNamespace):
    """Simple namespace mimicking `generate` return when `return_dict_in_generate=True`."""

    def __init__(self, sequences, scores):
        super().__init__(sequences=sequences, scores=scores)


class DummyModel:
    """Mimics minimal causal‑LM interface needed by LMPipeline."""

    def __init__(self):
        self.config = types.SimpleNamespace(name_or_path="dummy", num_hidden_layers=6)
        self.device = "cpu"

    # `to` just returns self so calls chain nicely
    def to(self, *_, **__):
        return self

    def generate(self, *_, **kwargs):
        batch = kwargs["input_ids"] if "input_ids" in kwargs else kwargs["input_ids"]
        batch_size = batch.shape[0]
        max_new = kwargs.get("max_new_tokens", 3)
        # sequences: pad_token_id + incremental ints
        seqs = torch.arange(1, max_new + 1).repeat(batch_size, 1)
        # fake logits: (batch, steps, vocab) where vocab=26 (a‑z)
        scores = [torch.rand(batch_size, 26) for _ in range(max_new)] if kwargs.get("output_scores") else None
        return DummyGenerateOutput(seqs, scores)

    # Needed by `prepare_inputs_for_generation` when position_ids=True
    def prepare_inputs_for_generation(self, input_ids, attention_mask):
        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)
        return {"position_ids": pos}


# ---------------------------------------------------------------------------
# Pytest fixtures & monkeypatches
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_hf(monkeypatch):
    """Patch HF factory funcs to return dummy stubs."""

    from transformers import AutoTokenizer, AutoModelForCausalLM

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: DummyTokenizer())
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **k: DummyModel())
    yield  # test runs


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

from neural.pipeline import LMPipeline, _infer_device_and_dtype  # noqa: E402 (import after patch)


def test_infer_device_and_dtype_cpu():
    device, dtype = _infer_device_and_dtype("cpu", None)
    assert device == "cpu"
    assert dtype == torch.float32


def test_pipeline_setup():
    pipe = LMPipeline("dummy‑model", max_new_tokens=2)
    # tokenizer / model replaced by dummy
    assert isinstance(pipe.tokenizer, DummyTokenizer)
    assert pipe.model.config.num_hidden_layers == 6
    assert pipe.max_new_tokens == 2


def test_load_returns_device_tensors():
    pipe = LMPipeline("dummy")
    batch = pipe.load("ab")
    assert batch["input_ids"].device.type == "cpu"
    assert batch["attention_mask"].sum() == 2  # two non‑pad tokens


def test_generate_and_dump():
    pipe = LMPipeline("dummy", max_new_tokens=3)
    output = pipe.generate(["hello", "world"])
    assert set(output.keys()) == {"scores", "sequences", "string"}
    assert output["sequences"].shape == (2, 3)
    text = pipe.dump(output["sequences"])
    assert isinstance(text, list) and len(text) == 2


def test_get_num_layers():
    pipe = LMPipeline("dummy")
    assert pipe.get_num_layers() == 6