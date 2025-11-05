import wandb

api = wandb.Api()
run = api.run("/goodfire/spd/runs/qycz8idd")

run.logs
