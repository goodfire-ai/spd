# NeuronDB Server Setup Guide

This guide explains how to set up and run the PostgreSQL NeuronDB server for Llama-3.1 8B Instruct neuron descriptions on the SLURM cluster.

## Overview

NeuronDB is a PostgreSQL database containing 458,752 neuron descriptions for Llama-3.1 8B Instruct. It runs inside a container via Pyxis/SLURM and provides neuron labels for the attribution graph pipeline.

## Full Setup from Scratch

### Step 1: Clone the Observatory Repository

```bash
cd /mnt/polished-lake/home/ctigges/code/attribution-graphs
git clone https://github.com/TransluceAI/observatory.git observatory_repo
```

### Step 2: Create the Environment File

Create `observatory_repo/.env` with PostgreSQL credentials:

```bash
cat > observatory_repo/.env << 'EOF'
# API Keys (optional for NeuronDB, required for other features)
OPENAI_API_ORG=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
HF_TOKEN=

# PostgreSQL credentials for NeuronDB container
PG_USER=clarity
PG_PASSWORD=sysadmin
PG_HOST=PLACEHOLDER
PG_PORT=5432
PG_DATABASE=neurons
EOF
```

Note: `PG_HOST` will be updated in Step 7 after starting the server.

### Step 3: Install Python Dependencies

Install the neurondb package and its dependencies into your venv:

```bash
cd /mnt/polished-lake/home/ctigges/code/attribution-graphs
uv pip install -e observatory_repo/lib/util -e observatory_repo/lib/activations -e observatory_repo/lib/neurondb
```

Required packages installed:
- `neurondb` - Database interface
- `util` - Environment variable handling
- `activations` - Neuron activation utilities
- `sqlalchemy`, `psycopg2-binary`, `pgvector` - PostgreSQL drivers

### Step 4: Pull the Container Image

Pull the preloaded NeuronDB container and save it as a squashfs file:

```bash
srun --container-image=docker://mengkluce/neurondb-llama-3.1-8b-instruct:latest \
     --container-name=neurondb \
     --container-save=/tmp/neurondb.sqsh \
     echo "Container saved"
```

This downloads ~17GB and saves to `/tmp/neurondb.sqsh`. Takes 5-10 minutes.

### Step 5: Create the Startup Script

Create `/tmp/start_neurondb.sh`:

```bash
cat > /tmp/start_neurondb.sh << 'EOF'
#!/bin/bash
# Create required directories for PostgreSQL
mkdir -p /var/run/postgresql
chmod 775 /var/run/postgresql

# Set PATH to include PostgreSQL binaries
export PATH=/usr/lib/postgresql/16/bin:$PATH

echo "Starting PostgreSQL on $(hostname)..."
echo "Data directory: /var/lib/postgresql/data"
echo "Listening on: 0.0.0.0:5432"

# Start PostgreSQL (running as current user, not root)
exec postgres -D /var/lib/postgresql/data -h 0.0.0.0 -p 5432
EOF

chmod +x /tmp/start_neurondb.sh
```

### Step 6: Start the PostgreSQL Server

Run PostgreSQL as a SLURM job:

```bash
srun --partition=h200-reserved \
     --nodes=1 \
     --ntasks=1 \
     --container-image=/tmp/neurondb.sqsh \
     --no-container-remap-root \
     --container-mounts=/tmp:/tmp \
     --job-name=neurondb-server \
     bash /tmp/start_neurondb.sh
```

**Critical flags:**
- `--partition=h200-reserved` - Use compute nodes (avoids dev node conflicts)
- `--no-container-remap-root` - Run as your user (PostgreSQL refuses to run as root)
- `--container-mounts=/tmp:/tmp` - Mount /tmp so the startup script is accessible

Wait ~10 seconds for PostgreSQL to initialize. You should see:
```
Starting PostgreSQL on h200-reserved-145-XXX...
LOG:  database system is ready to accept connections
```

### Step 7: Update Configuration with Hostname

Find which node the job is running on:

```bash
squeue -u $USER -o "%.18i %.20j %.8T %.6D %R"
```

Example output:
```
JOBID                 NAME    STATE       TIME  NODES NODELIST(REASON)
12345       neurondb-server  RUNNING       0:15      1 h200-reserved-145-011
```

Update `observatory_repo/.env` with the actual hostname:

```bash
sed -i 's/PG_HOST=.*/PG_HOST=h200-reserved-145-011/' observatory_repo/.env
```

Or manually edit the file to set `PG_HOST=h200-reserved-145-011` (use your actual node).

### Step 8: Verify the Connection

Test the database connection:

```bash
.venv/bin/python -c "
import os
os.chdir('observatory_repo')

from neurondb.postgres import DBManager
from neurondb.schemas import SQLANeuron, SQLANeuronDescription
from sqlalchemy import func

db = DBManager.get_instance()
print('✓ Connected to NeuronDB!')

neuron_count = db.get([func.count(SQLANeuron.id)])[0][0]
print(f'✓ Total neurons: {neuron_count:,}')

# Test a sample query
results = db.get(
    [SQLANeuron.layer, SQLANeuron.neuron, SQLANeuronDescription.description],
    joins=[(SQLANeuronDescription, SQLANeuron.id == SQLANeuronDescription.neuron_id)],
    limit=1
)
if results:
    layer, neuron, desc = results[0]
    print(f'✓ Sample: Layer {layer}, Neuron {neuron}')
    print(f'  Description: {desc[:100]}...')
"
```

Expected output:
```
✓ Connected to NeuronDB!
✓ Total neurons: 458,752
✓ Sample: Layer 2, Neuron 4217
  Description: Tokens containing "Hari" and its variations...
```

---

## Quick Restart (Recommended)

The container image and startup scripts are already set up in permanent locations. Use these commands:

### 1. Start the Server

```bash
cd /mnt/polished-lake/home/ctigges/code/attribution-graphs
sbatch scripts/start_neurondb.sh
```

This submits a background job that runs for 4 hours. The container image is at `/mnt/polished-lake/home/ctigges/containers/neurondb.sqsh`.

### 2. Find the Node and Update Config

```bash
# Wait a few seconds for the job to start, then find the node
squeue -u $USER -o "%.18i %.20j %.8T %.6D %R"

# Example output:
# JOBID                 NAME    STATE  NODES NODELIST(REASON)
# 59187      neurondb-server  RUNNING      1 h200-reserved-145-002

# Update BOTH .env files with the actual node hostname
sed -i 's/PG_HOST=.*/PG_HOST=h200-reserved-145-XXX/' .env
sed -i 's/PG_HOST=.*/PG_HOST=h200-reserved-145-XXX/' observatory_repo/.env
```

**Important:** Both `.env` (in repo root) and `observatory_repo/.env` must be updated. The root `.env` is used by SLURM jobs via `export $(grep -v '^#' .env | xargs)`.

### 3. Verify PostgreSQL Started

```bash
# Check the log for "database system is ready to accept connections"
cat logs/neurondb_*.err | tail -5
```

### 4. Test Connection

```bash
.venv/bin/python -c "
import os; os.chdir('observatory_repo')
from dotenv import load_dotenv; load_dotenv()
from neurondb.postgres import DBManager
from neurondb.schemas import SQLANeuron
from sqlalchemy import func
db = DBManager.get_instance()
print(f'Connected! Neurons: {db.get([func.count(SQLANeuron.id)])[0][0]:,}')
"
```

Expected output: `Connected! Neurons: 458,752`

---

## Quick Restart (Legacy - /tmp locations)

If using the old /tmp-based setup:

### 1. Verify Prerequisites

```bash
ls -lh /tmp/neurondb.sqsh      # Should be ~17GB
ls -l /tmp/start_neurondb.sh   # Should exist and be executable
```

### 2. Start the Server

```bash
srun --partition=h200-reserved \
     --nodes=1 \
     --ntasks=1 \
     --container-image=/tmp/neurondb.sqsh \
     --no-container-remap-root \
     --container-mounts=/tmp:/tmp \
     --job-name=neurondb-server \
     bash /tmp/start_neurondb.sh
```

### 3. Update PG_HOST

```bash
# Find the node
squeue -u $USER -o "%.18i %.20j %.8T %.6D %R"

# Update .env (replace h200-reserved-145-XXX with actual node)
sed -i 's/PG_HOST=.*/PG_HOST=h200-reserved-145-XXX/' observatory_repo/.env
```

### 4. Test Connection

```bash
.venv/bin/python -c "
import os; os.chdir('observatory_repo')
from neurondb.postgres import DBManager
from neurondb.schemas import SQLANeuron
from sqlalchemy import func
db = DBManager.get_instance()
print(f'Connected! Neurons: {db.get([func.count(SQLANeuron.id)])[0][0]:,}')
"
```

---

## Troubleshooting

### "Connection refused" Error

```
psycopg2.OperationalError: connection to server at "h200-xxx" failed: Connection refused
```

**Solutions:**
1. Check if PostgreSQL is running: `squeue -u $USER`
2. Verify `PG_HOST` in `observatory_repo/.env` matches the SLURM job node
3. Restart the server if the job ended

### "root execution of PostgreSQL server is not permitted"

**Cause:** Running without `--no-container-remap-root` flag

**Solution:** Always include `--no-container-remap-root` in the srun command

### "could not create lock file /var/run/postgresql/.s.PGSQL.5432.lock"

**Cause:** `/var/run/postgresql` doesn't exist in the container

**Solution:** Ensure the startup script creates this directory. Recreate `/tmp/start_neurondb.sh` if needed.

### "Requested nodes are busy"

**Cause:** SLURM job queue congestion

**Solutions:**
1. Use `--partition=h200-reserved` to access more nodes
2. Wait for blocking jobs to finish
3. Check for your own blocking jobs: `squeue -u $USER`

### Container Image Not Found

```
srun: error: container image not found: /tmp/neurondb.sqsh
```

**Solution:** Re-pull the container (Step 4 above):
```bash
srun --container-image=docker://mengkluce/neurondb-llama-3.1-8b-instruct:latest \
     --container-name=neurondb \
     --container-save=/tmp/neurondb.sqsh \
     echo "Container pulled and saved"
```

### Python Import Errors

```
ModuleNotFoundError: No module named 'neurondb'
```

**Solution:** Reinstall the packages (Step 3 above):
```bash
uv pip install -e observatory_repo/lib/util -e observatory_repo/lib/activations -e observatory_repo/lib/neurondb
```

---

## Job Management

### Check Job Status
```bash
squeue -u $USER
```

### Stop the Server
```bash
scancel <job_id>
```

### View Available Partitions
```bash
sinfo -o "%P %a %l %D %N"
```

---

## File Locations Summary

| File | Purpose |
|------|---------|
| `/mnt/polished-lake/home/ctigges/containers/neurondb.sqsh` | Container image (17GB, permanent location) |
| `scripts/start_neurondb.sh` | SLURM job script to start the server |
| `scripts/neurondb_init.sh` | PostgreSQL startup script (runs inside container) |
| `.env` | Root environment config (API keys + database connection) |
| `observatory_repo/.env` | Observatory environment config (database connection) |
| `observatory_repo/lib/neurondb/` | Python package source |

**Note:** Both `.env` files need `PG_HOST` updated when starting the server. The root `.env` is used by SLURM jobs.

---

## Network Architecture

```
┌─────────────────────────────────────┐
│  Login/Dev Node                     │
│  (where you run Python scripts)     │
│                                     │
│  observatory_repo/.env:             │
│    PG_HOST=h200-reserved-145-XXX    │
└──────────────┬──────────────────────┘
               │
               │ TCP:5432
               │
┌──────────────▼──────────────────────┐
│  Compute Node (h200-reserved-145-XXX)│
│  ┌────────────────────────────────┐ │
│  │ SLURM Job Container            │ │
│  │ (from /tmp/neurondb.sqsh)      │ │
│  │                                │ │
│  │  ┌──────────────────────────┐  │ │
│  │  │ PostgreSQL 16 + pgvector │  │ │
│  │  │                          │  │ │
│  │  │ Database: neurons        │  │ │
│  │  │ 458,752 neuron records   │  │ │
│  │  │ with descriptions        │  │ │
│  │  └──────────────────────────┘  │ │
│  └────────────────────────────────┘ │
└─────────────────────────────────────┘
```

---

## Database Schema

Key tables in NeuronDB:

| Table | Description |
|-------|-------------|
| `language_models` | Model metadata (e.g., "llama-3.1-8b-instruct") |
| `neurons` | Neuron locations (layer, neuron index) |
| `neuron_descriptions` | Text descriptions with vector embeddings |
| `neuron_quantiles` | Activation statistics |
| `neuron_exemplars` | Example inputs that activate neurons |

---

## Python Usage Examples

### Basic Query

```python
import os
os.chdir('observatory_repo')

from neurondb.postgres import DBManager
from neurondb.schemas import SQLANeuron, SQLANeuronDescription

db = DBManager.get_instance()

# Get descriptions for specific neurons
results = db.get(
    [SQLANeuron.layer, SQLANeuron.neuron, SQLANeuronDescription.description],
    joins=[(SQLANeuronDescription, SQLANeuron.id == SQLANeuronDescription.neuron_id)],
    filter=(SQLANeuron.layer == 10),
    limit=100
)

for layer, neuron, description in results:
    print(f"Layer {layer}, Neuron {neuron}: {description}")
```

### Query by Layer-Neuron Tuples

```python
# Query specific neurons by (layer, neuron) pairs
neuron_list = [(5, 100), (10, 200), (15, 300)]

results = db.get(
    [SQLANeuron.layer, SQLANeuron.neuron, SQLANeuronDescription.description],
    joins=[(SQLANeuronDescription, SQLANeuron.id == SQLANeuronDescription.neuron_id)],
    layer_neuron_tuples=neuron_list
)
```

---

## Container Details

- **Source Image:** `mengkluce/neurondb-llama-3.1-8b-instruct:latest` (Docker Hub)
- **Local File:** `/tmp/neurondb.sqsh` (squashfs format, ~17GB)
- **PostgreSQL Version:** 16.4
- **Extensions:** pgvector (for vector similarity search)
- **Credentials:** user=`clarity`, password=`sysadmin`
- **Database:** `neurons`
- **Port:** 5432
