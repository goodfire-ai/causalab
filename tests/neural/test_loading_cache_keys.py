"""Equivalent call spellings load once; distinct realizations stay distinct."""

from types import SimpleNamespace

import pytest

from causalab.neural.engines.pytorch_hooks import loading as hooks
from causalab.neural.engines.nnsight_tracing import loading as tracing
from causalab.neural.shared.devices import DeviceMap

pytestmark = pytest.mark.unit


@pytest.fixture(params=[hooks, tracing], ids=["hooks", "tracing"])
def loader(request, monkeypatch):
    module = request.param
    calls = []

    def construct(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            config=object(),
            tokenizer=SimpleNamespace(pad_token=None, eos_token="eos"),
            eval=lambda: None,
            requires_grad_=lambda value: None,
        )

    # the registry entry the loaders read the tower depth off (`DeviceMap.parse`)
    monkeypatch.setattr(
        module, "model_info_from_hf_config", lambda *args: SimpleNamespace(num_layers=1)
    )
    monkeypatch.setattr(module, "register_model", lambda info: None)
    monkeypatch.setattr(module, "bind_kernel_path", lambda *args, **kwargs: None)
    if module is hooks:
        import transformers

        monkeypatch.setattr(module, "load_pretrained", construct)
        monkeypatch.setattr(module, "_bitsandbytes_config", lambda config: config)
        # the quantized path reads the checkpoint's config for its depth
        # before bitsandbytes builds the model; the stand-in has no checkpoint
        monkeypatch.setattr(module, "_config_of", lambda *args: object())
        # the placement is read off a real model's parameters through the
        # family's tree; the stand-in has none, so it is the requested device
        monkeypatch.setattr(
            module,
            "_placement_of",
            lambda model, requested=None, **_: DeviceMap.parse(requested or "cpu", 1),
        )
        monkeypatch.setattr(
            transformers.AutoModelForCausalLM, "from_pretrained", construct
        )
        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: SimpleNamespace(pad_token=None, eos_token="eos"),
        )
    else:
        import nnsight.modeling.transformers

        monkeypatch.setattr(
            nnsight.modeling.transformers, "TransformersModel", construct
        )
        monkeypatch.setattr(module, "torch_module", lambda model: model)
    # an isolated cache: the session's fixtures hold bundles out of the real one
    monkeypatch.setattr(module, "load_model", module.load_model.renewed())
    yield module, calls


def test_equivalent_calls_share_one_bundle(loader):
    module, calls = loader
    load = module.load_model
    first = load("example")
    assert load(key="example", revision="main") is first
    assert load("example", "main", device="cpu", dtype="fp32") is first
    options = {"attn_implementation": "eager" if module is hooks else None}
    if module is hooks:
        options["quantization"] = None
    assert load("example", **options) is first
    assert len(calls) == 1
    assert load.cache_info().maxsize == 4
    load.cache_clear()
    assert load("example") is not first
    assert len(calls) == 2


@pytest.mark.parametrize(
    "difference",
    [
        {"revision": "other"},
        {"dtype": "bf16"},
        {"device": "cuda:0"},
        {"attn_implementation": "sdpa"},
    ],
)
def test_distinct_realizations_do_not_collide(loader, difference):
    module, calls = loader
    first = module.load_model("example")
    assert module.load_model("example", **difference) is not first
    assert len(calls) == 2


def test_none_is_not_eager_and_quantization_order_is_irrelevant(loader):
    module, calls = loader
    assert module.load_model(
        "example", attn_implementation=None
    ) is not module.load_model("example", attn_implementation="eager")
    if module is hooks:
        block = {"method": "bitsandbytes", "scheme": "nf4"}
        first = module.load_model("example", quantization=block)
        reordered = {"scheme": "nf4", "method": "bitsandbytes"}
        assert module.load_model("example", quantization=reordered) is first
        assert (
            module.load_model("example", quantization={"scheme": "int8"}) is not first
        )
        quantized = [
            kw["quantization_config"] for _, kw in calls if "quantization_config" in kw
        ]
        assert quantized == [block, {"scheme": "int8"}], (
            "the constructor receives the document's block, not the key"
        )
