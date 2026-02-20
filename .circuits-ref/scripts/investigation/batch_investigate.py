#!/usr/bin/env python3
"""Run neuron investigations in parallel batches."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neuron_scientist.agent import NeuronScientist


async def investigate_neuron(neuron_id: str) -> dict:
    """Investigate a single neuron."""
    print(f"\n{'='*60}")
    print(f"Starting investigation: {neuron_id}")
    print(f"{'='*60}")

    try:
        scientist = NeuronScientist(
            neuron_id=neuron_id,
            edge_stats_path=Path('data/edge_stats_medical.json'),
            output_dir=Path('outputs/investigations'),
            model='opus',
        )

        result = await scientist.investigate(max_experiments=100)

        print(f"\n✓ {neuron_id} complete: {result.total_experiments} experiments, {result.confidence:.0%} confidence")
        return {
            "neuron_id": neuron_id,
            "success": True,
            "experiments": result.total_experiments,
            "confidence": result.confidence,
        }
    except Exception as e:
        print(f"\n✗ {neuron_id} failed: {e}")
        return {
            "neuron_id": neuron_id,
            "success": False,
            "error": str(e),
        }


async def run_batch(neuron_ids: list, batch_size: int = 4):
    """Run investigations in batches."""
    results = []

    for i in range(0, len(neuron_ids), batch_size):
        batch = neuron_ids[i:i + batch_size]
        print(f"\n{'#'*60}")
        print(f"BATCH {i//batch_size + 1}: {batch}")
        print(f"{'#'*60}")

        # Run batch in parallel
        batch_results = await asyncio.gather(
            *[investigate_neuron(nid) for nid in batch],
            return_exceptions=True
        )

        for r in batch_results:
            if isinstance(r, Exception):
                results.append({"error": str(r)})
            else:
                results.append(r)

        # Brief pause between batches to let GPU memory settle
        if i + batch_size < len(neuron_ids):
            print("\nPausing 10s before next batch to let GPU memory settle...")
            import gc
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
            await asyncio.sleep(10)

    return results


async def main():
    # 16 diverse neurons to investigate
    neurons = [
        'L31/N127',      # patient/patients promoter (MEDICAL)
        'L31/N5335',     # Food and flavor terms promoter (FOOD)
        'L31/N12183',    # animal/pet term promoter (ANIMAL)
        'L31/N11155',    # basic color-word promoter (COLOR)
        'L20/N11869',    # Nationality-adjective promoter (GEOGRAPHIC)
        'L31/N6271',     # protein-related token promoter (BIOLOGY)
        'L31/N13610',    # Promotes "what" across casings (INTERROGATIVE)
        'L31/N5801',     # Non-English first-person pronoun promoter (PRONOUN)
        'L15/N6126',     # thank-you context (GRATITUDE)
        'L30/N11230',    # apology "sorry to hear" trigger (APOLOGY)
        'L25/N1252',     # Congratulatory-context (CELEBRATION)
        'L31/N13282',    # Suppresses Sure/yes (AFFIRMATIVE)
        'L29/N142',      # year-prefix and delimiter (TEMPORAL)
        'L31/N8508',     # Multilingual plural-suffix promoter (GRAMMAR)
        'L31/N10770',    # Curly-brace-number/template (NUMERIC)
        'L31/N5722',     # Suppresses capitalized function-words (CASING)
    ]

    print(f"Investigating {len(neurons)} neurons in batches of 2...")
    results = await run_batch(neurons, batch_size=2)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]

    print(f"\nSuccessful: {len(successes)}/{len(results)}")
    for r in successes:
        print(f"  ✓ {r['neuron_id']}: {r['experiments']} experiments, {r['confidence']:.0%} confidence")

    if failures:
        print(f"\nFailed: {len(failures)}")
        for r in failures:
            print(f"  ✗ {r.get('neuron_id', 'unknown')}: {r.get('error', 'unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())
