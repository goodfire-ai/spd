#!/usr/bin/env python3
"""Re-label neurons with truncated labels after regex fix."""

import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent))
from interactive_labeling import InteractiveLabeler


async def relabel_neurons():
    # Load neurons to relabel
    with open('data/neurons_to_relabel.json') as f:
        neurons_to_relabel = set(json.load(f))

    print(f"Neurons to relabel: {len(neurons_to_relabel)}")

    # Load edge stats to get profiles
    with open('data/fineweb_50k_edge_stats_enriched.json') as f:
        stats = json.load(f)

    profiles_list = stats.get('profiles', [])
    profiles = {p['neuron_id']: p for p in profiles_list}

    # Get profiles for neurons to relabel
    relabel_profiles = [profiles[nid] for nid in neurons_to_relabel if nid in profiles]
    print(f"Found profiles for: {len(relabel_profiles)}")

    # Initialize labeler
    labeler = InteractiveLabeler(
        edge_stats_path=Path('data/fineweb_50k_edge_stats_enriched.json'),
        db_path=Path('data/neuron_function_db_full.json'),
        state_path=Path('data/.labeling_state_relabel.json'),
        model='gpt-4.1-mini',
        batch_size=800,
        label_pass='output',
    )

    client = AsyncOpenAI()

    print(f"\nRe-labeling {len(relabel_profiles)} neurons...")
    start_time = time.time()

    batch_size = 800
    total_fixed = 0

    for i in range(0, len(relabel_profiles), batch_size):
        batch = relabel_profiles[i:i + batch_size]

        # Build prompts
        prompts = [labeler.build_prompt_for_neuron(p) for p in batch]

        # Call LLM in parallel
        async def call_llm(prompt):
            response = await client.chat.completions.create(
                model='gpt-4.1-mini',
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3,
            )
            return response.choices[0].message.content

        tasks = [call_llm(p) for p in prompts]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Parse and save
        for profile, response in zip(batch, responses):
            if isinstance(response, Exception):
                print(f"Error for {profile['neuron_id']}: {response}")
                continue

            nid = profile['neuron_id']
            parsed = labeler.parse_structured_response(response)

            # Update in database
            neuron_func = labeler.interpreter.db.get(nid)
            if neuron_func:
                neuron_func.function_label = parsed["short_label"]
                neuron_func.function_description = parsed["output_function"]
                neuron_func.function_type = parsed["function_type"]
                neuron_func.interpretability = parsed["interpretability"]
                labeler.interpreter.db.set(neuron_func)
                total_fixed += 1

        labeler.interpreter.db.save()

        elapsed = time.time() - start_time
        rate = (i + len(batch)) / elapsed * 60
        print(f"Batch {i//batch_size + 1}: {i + len(batch)}/{len(relabel_profiles)} ({rate:.0f}/min)")

    print(f"\nDone! Fixed {total_fixed} neurons in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    asyncio.run(relabel_neurons())
