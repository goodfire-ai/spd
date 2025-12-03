import * as api from "./lib/api";

async function main() {
    console.log("loading run");
    await api.loadRun("goodfire/spd/lxs77xye");
    console.log("run loaded");

    console.log("getting subcomponent activation contexts");
    await api.getSubcomponentActivationContexts({
        n_batches: 4,
        batch_size: 2,
        n_tokens_either_side: 10,
        importance_threshold: 0.0,
        topk_examples: 40,
        separation_tokens: 0,
    });
    console.log("subcomponent activation contexts got");
}

main();
