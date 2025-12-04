# Load the model https://wandb.ai/goodfire/spd/runs/33n6xjjt using SPDRunInfo.from_path

import time

from spd.models.component_model import SPDRunInfo

start_time = time.time()
run_info = SPDRunInfo.from_path("wandb:goodfire/spd/runs/33n6xjjt")
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
