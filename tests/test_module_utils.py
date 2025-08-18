from torch import nn

from spd.utils.module_utils import get_nested_module_attr


def test_get_nested_module_attr():
    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 10)
            self.linear2 = nn.Linear(10, 10)

    module = TestModule()
    assert get_nested_module_attr(module, "linear1.weight.data").shape == (10, 10)
    assert get_nested_module_attr(module, "linear2.weight.data").shape == (10, 10)
