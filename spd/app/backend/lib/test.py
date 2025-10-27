import time
from collections.abc import Callable

from spd.app.backend.lib.activation_contexts_common import ActivationsData
from spd.app.backend.lib.activation_contexts_v0 import get_topk_by_subcomponent_v0
from spd.app.backend.lib.activation_contexts_v1 import get_topk_by_subcomponent_v1
from spd.app.backend.lib.activation_contexts_v2 import get_topk_by_subcomponent_v2
from spd.app.backend.lib.activation_contexts_v3 import get_topk_by_subcomponent_v3
from spd.app.backend.lib.activation_contexts_v4 import get_topk_by_subcomponent_v4
from spd.app.backend.lib.activation_contexts_v5 import get_topk_by_subcomponent_v5
from spd.app.backend.lib.activation_contexts_v6 import get_topk_by_subcomponent_v6
from spd.app.backend.lib.activation_contexts_v7 import get_topk_by_subcomponent_v7
from spd.app.backend.services.run_context_service import RunContextService
from spd.utils.distributed_utils import get_device

rcs = RunContextService()
rcs.load_run("goodfire/spd/lxs77xye")
run_context = rcs.train_run_context
assert run_context is not None

device = get_device()

def test(f: Callable[..., ActivationsData]) -> float:
    start = time.time()
    f(
        run_context=run_context,
        importance_threshold=0.0,
        n_batches=4,
        n_tokens_either_side=10,
        batch_size=4,
        device=device,
    )
    return time.time() - start


# print(f"V0: {test(get_topk_by_subcomponent_v0)}")
# print(f"V1: {test(get_topk_by_subcomponent_v1)}")
# print(f"V2: {test(get_topk_by_subcomponent_v2)}")
# print(f"V3: {test(get_topk_by_subcomponent_v3)}")
# print(f"V4: {test(get_topk_by_subcomponent_v4)}")
print(f"V5: {test(get_topk_by_subcomponent_v5)}")
# print(f"V6: {test(get_topk_by_subcomponent_v6)}")
print(f"V7: {test(get_topk_by_subcomponent_v7)}")