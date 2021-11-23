import importlib
from types import ModuleType
from typing import Union
from unittest import mock

import pytest

TO_MOCK = ["elasticsearch", "torch", "transformers"]


def import_mock(name: str, *args, **kwargs) -> Union[mock.Mock, ModuleType]:
    """Returns mocked module if module name is in TO_MOCK.

    Args:
        name: Module name

    Returns:
        Returns mocked module if module name is in TO_MOCK, otherwise imports
        module.
    """
    if any(skip in name for skip in TO_MOCK):
        return mock.Mock()
    return importlib.__import__(name, *args, **kwargs)


@pytest.helpers.register
def mock_expensive_imports() -> mock._patch:
    """Mocks import all imports that are in the list `to_mock`.

    Returns:
        mock.patch with patched builtin import function.
    """

    return mock.patch("builtins.__import__", side_effect=import_mock)
