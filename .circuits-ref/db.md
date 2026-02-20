Summary

  NeuronDB is now running in this folder. Here's what was set up:

  Database Server (running as SLURM job 23f3aa):
  - PostgreSQL 16 with pgvector extension
  - Contains 458,752 neuron descriptions for Llama-3.1 8B Instruct
  - Running on compute node h200-dev-145-045:5432

  Python Package:
  - Installed neurondb, util, and activations from observatory_repo/lib/

  Configuration (observatory_repo/.env):
  PG_HOST=h200-dev-145-045
  PG_PORT=5432
  PG_USER=clarity
  PG_PASSWORD=sysadmin
  PG_DATABASE=neurons

  Usage Example:
  import os
  os.chdir('observatory_repo')

  from neurondb.postgres import DBManager
  from neurondb.schemas import SQLANeuron, SQLANeuronDescription

  db = DBManager.get_instance()
  results = db.get(
      [SQLANeuron.layer, SQLANeuron.neuron, SQLANeuronDescription.description],
      joins=[(SQLANeuronDescription, SQLANeuron.id == SQLANeuronDescription.neuron_id)],
      limit=10
  )

  Note: The PostgreSQL server is running as a background SLURM job. When the job ends, you'll need to restart it
  with:
  srun --container-name=neurondb --no-container-remap-root --container-mounts=/tmp:/tmp bash