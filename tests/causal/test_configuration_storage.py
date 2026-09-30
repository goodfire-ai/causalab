"""Configuration capture copies stored state without invoking user accessors."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from causalab.causal import CausalModel, DefinitionError, Dom, V, mechanism
from causalab.causal.compiler import ConfigurationCopier

pytestmark = pytest.mark.unit


def test_private_dataclass_slots_preserve_model_outcome_and_snapshot():
    @dataclass
    class Config:
        __slots__ = ("__offset",)
        __offset: list

        @property
        def offset(self):
            return getattr(self, "_Config__offset", [0])[0]

    config = Config([10])

    @mechanism
    def equations(x: Dom([1, 2])):
        result = V(x + config.offset, domain=Dom(int))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    config._Config__offset[0] = 30
    model.definition.environment["config"]._Config__offset[0] = 40
    assert model.new_trace({"x": 1})["result"] == 11


def test_descriptor_setters_are_not_reapplied_during_model_capture():
    class Twice:
        def __set_name__(self, owner, name):
            self.name = name

        def __get__(self, instance, owner=None):
            if instance is None:
                return self
            return instance.__dict__[self.name]

        def __set__(self, instance, value):
            instance.__dict__[self.name] = value * 2

    @dataclass
    class Config:
        offset: int = Twice()

    config = Config(5)

    @mechanism
    def equations(x: Dom([1, 2])):
        result = V(x + config.offset, domain=Dom(int))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    assert config.offset == 10
    assert CausalModel(equations).new_trace({"x": 1})["result"] == 11


def test_dataclass_capture_does_not_call_field_descriptor_accessors():
    calls = []

    class Watched:
        def __get__(self, instance, owner=None):
            if instance is None:
                return self
            calls.append("get")
            return instance.__dict__["offset"]

        def __set__(self, instance, value):
            calls.append("set")
            instance.__dict__["offset"] = value

    @dataclass
    class Config:
        offset: list = Watched()

    config = Config([10])
    calls.clear()
    copied = ConfigurationCopier()(config)
    assert calls == []
    assert copied.__dict__["offset"] == [10]
    assert copied.__dict__["offset"] is not config.__dict__["offset"]


def test_dictionary_storage_bypasses_an_overridden_dict_property():
    @dataclass
    class Base:
        offset: list

    class Config(Base):
        @property
        def __dict__(self):
            raise AssertionError("user __dict__ accessor must not run")

    config = Config([10])
    copied = ConfigurationCopier()(config)
    assert type(copied) is Config
    assert copied.offset == [10]
    config.offset[0] = 20
    assert copied.offset == [10]


def test_hidden_dictionary_storage_is_rejected_without_calling_its_property():
    @dataclass
    class Config:
        offset: list

        @property
        def __dict__(self):
            raise AssertionError("user __dict__ accessor must not run")

    with pytest.raises(DefinitionError, match="hidden __dict__ storage"):
        ConfigurationCopier()(Config([10]))


def test_dataclass_slot_storage_bypasses_attribute_overrides():
    @dataclass(slots=True)
    class Config:
        offset: list

        def __getattribute__(self, name):
            if name == "offset":
                raise AssertionError("user attribute getter must not run")
            return object.__getattribute__(self, name)

        def __setattr__(self, name, value):
            if name == "offset":
                raise AssertionError("user attribute setter must not run")
            object.__setattr__(self, name, value)

    config = object.__new__(Config)
    descriptor = Config.__dict__["offset"]
    descriptor.__set__(config, [10])
    copied = ConfigurationCopier()(config)
    assert descriptor.__get__(copied, Config) == [10]
    assert descriptor.__get__(copied, Config) is not descriptor.__get__(config, Config)


def test_inherited_private_and_shadowed_slots_keep_separate_storage():
    class Base:
        __slots__ = ("__offset", "shared")

    @dataclass
    class Config(Base):
        __slots__ = ("__offset", "shared", "__dict__")

    config = Config()
    Base.__dict__["_Base__offset"].__set__(config, [1])
    Base.__dict__["shared"].__set__(config, [2])
    Config.__dict__["_Config__offset"].__set__(config, [3])
    Config.__dict__["shared"].__set__(config, [4])
    config.extra = [5]
    copied = ConfigurationCopier()(config)
    for owner, name, expected in (
        (Base, "_Base__offset", [1]),
        (Base, "shared", [2]),
        (Config, "_Config__offset", [3]),
        (Config, "shared", [4]),
    ):
        descriptor = owner.__dict__[name]
        assert descriptor.__get__(copied, Config) == expected
        assert descriptor.__get__(copied, Config) is not descriptor.__get__(
            config, Config
        )
    assert copied.extra == [5]
    assert copied.extra is not config.extra


def test_uninitialized_slots_remain_uninitialized():
    @dataclass(slots=True)
    class Config:
        offset: list
        absent: int = field(init=False)

    copied = ConfigurationCopier()(Config([10]))
    assert copied.offset == [10]
    assert not hasattr(copied, "absent")


def test_frozen_slotted_inherited_dataclass_is_copied_without_initialization():
    @dataclass(frozen=True, slots=True)
    class Base:
        offset: list

    @dataclass(frozen=True, slots=True)
    class Config(Base):
        scale: list

        def __post_init__(self):
            self.offset[0] *= self.scale[0]

    config = Config([5], [2])
    copied = ConfigurationCopier()(config)
    assert type(copied) is Config
    assert copied.offset == [10]
    assert copied.scale == [2]
    assert copied.offset is not config.offset


def test_namespace_dataclass_keeps_exact_class_methods_and_owned_values():
    @dataclass
    class Config(SimpleNamespace):
        offset: list

        def apply(self, value):
            return value + self.offset[0]

    config = Config([10])

    @mechanism
    def equations():
        result = V((isinstance(config, Config), config.apply(1)), domain=Dom(tuple))
        raw_input = V("input")  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    config.offset[0] = 20
    assert model.new_trace()["result"] == (True, 11)


def test_namespace_subclass_bypasses_custom_constructor_and_has_slots():
    class Config(SimpleNamespace):
        __slots__ = ("__offset",)

        def __new__(cls, token):
            assert token == "initial"
            return SimpleNamespace.__new__(cls)

        def __init__(self, token):
            self._Config__offset = [10]
            self.shared = self._Config__offset

    config = Config("initial")
    copied = ConfigurationCopier()(config)
    assert type(copied) is Config
    assert copied.shared is copied._Config__offset
    assert copied.shared == [10]
    assert copied.shared is not config.shared


@pytest.mark.parametrize("kind", ["dataclass", "namespace"])
def test_self_references_and_dictionary_storage_aliases_survive(kind):
    @dataclass
    class Config:
        offset: list

    config = Config([10]) if kind == "dataclass" else SimpleNamespace(offset=[10])
    config.owner = config
    config.storage = config.__dict__
    copied = ConfigurationCopier()(config)
    assert copied.owner is copied
    assert copied.storage is copied.__dict__
    assert copied.offset is not config.offset


def test_shared_readonly_dictionary_is_rejected_instead_of_losing_its_alias():
    config = SimpleNamespace(offset=[10])
    values = {"storage": config.__dict__, "config": config}
    with pytest.raises(DefinitionError, match="shared read-only __dict__"):
        ConfigurationCopier()(values)


def test_domain_subclass_copies_stored_state_without_descriptor_setters():
    class ConfiguredDomain(Dom):
        @property
        def values(self):
            return self.__dict__["values"]

        @values.setter
        def values(self, value):
            self.__dict__["values"] = tuple(item * 2 for item in value)

    domain = ConfiguredDomain([1, 2])
    copied = ConfigurationCopier()(domain)
    assert type(copied) is ConfiguredDomain
    assert copied.require_enumerated() == (2, 4)


def test_storage_copy_preserves_callable_aware_capture():
    @dataclass
    class Config:
        offset: list
        helper: object

    offset = [10]

    def helper(value):
        return value + offset[0]

    config = Config(offset, helper)
    copied = ConfigurationCopier()(config)
    offset[0] = 20
    assert copied.helper(1) == 11
    copied.offset[0] = 30
    assert copied.helper(1) == 31
