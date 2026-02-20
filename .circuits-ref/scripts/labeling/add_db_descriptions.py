#!/usr/bin/env python
"""Add neurondb descriptions to MVP labels for comparison."""
import json
import os
import sys

# Setup path
sys.path.insert(0, 'observatory_repo/lib/neurondb')
sys.path.insert(0, 'observatory_repo/lib/util')

os.chdir('observatory_repo')
from dotenv import load_dotenv

load_dotenv()

from neurondb.postgres import DBManager
from neurondb.schemas import SQLANeuron, SQLANeuronDescription

# Load MVP labels
os.chdir('..')
with open('data/mvp_labels_compositional.json') as f:
    data = json.load(f)

# Extract (layer, neuron) tuples
neuron_list = []
for nid in data['labels']:
    layer = data['labels'][nid]['layer']
    neuron = data['labels'][nid]['neuron']
    neuron_list.append((layer, neuron))

print(f"Querying {len(neuron_list)} neurons from database...")

# Query database
os.chdir('observatory_repo')
db = DBManager.get_instance()

results = db.get(
    [SQLANeuron.layer, SQLANeuron.neuron, SQLANeuronDescription.description],
    joins=[(SQLANeuronDescription, SQLANeuron.id == SQLANeuronDescription.neuron_id)],
    layer_neuron_tuples=neuron_list
)

# Build lookup
db_descriptions = {}
for layer, neuron, desc in results:
    nid = f"L{layer}/N{neuron}"
    db_descriptions[nid] = desc

print(f"Found {len(db_descriptions)} descriptions in database")

# Add to labels
os.chdir('..')
for nid in data['labels']:
    if nid in db_descriptions:
        data['labels'][nid]['db_description'] = db_descriptions[nid]
    else:
        data['labels'][nid]['db_description'] = "(not in database)"

# Save
with open('data/mvp_labels_compositional.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Added db_description to all neurons")

# Show comparison examples
print("\n" + "="*80)
print("COMPARISON: Database vs Compositional Labels")
print("="*80 + "\n")

examples = ['L0/N857', 'L2/N12082', 'L9/N12312', 'L15/N11853', 'L28/N447', 'L31/N311']
for nid in examples:
    if nid in data['labels']:
        label = data['labels'][nid]
        print(f"{'─'*80}")
        print(f"NEURON: {nid}")
        print(f"{'─'*80}")
        db_desc = label.get('db_description', 'N/A')
        print("\n📚 DATABASE DESCRIPTION:")
        print(f"   {db_desc[:300]}{'...' if len(db_desc) > 300 else ''}")
        print("\n🔬 COMPOSITIONAL LABEL:")
        print(f"   {label.get('complete_function', 'N/A')}")
        print()
