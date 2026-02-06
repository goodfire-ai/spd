"""API endpoint tests for spd.app.backend.server.

These tests bypass slow operations (W&B model loading, large data loaders) by:
1. Manually constructing app state with a fresh randomly-initialized model
2. Using small data splits and short sequences
3. Using an in-memory SQLite database
"""

import json
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from spd.app.backend.compute import get_sources_by_target
from spd.app.backend.database import PromptAttrDB
from spd.app.backend.routers import graphs as graphs_router
from spd.app.backend.routers import prompts as prompts_router
from spd.app.backend.routers import runs as runs_router
from spd.app.backend.server import app
from spd.app.backend.state import HarvestCache, RunState, StateManager
from spd.configs import Config, LMTaskConfig, ModulePatternInfoConfig, ScheduleConfig
from spd.models.component_model import ComponentModel
from spd.pretrain.models.gpt2_simple import GPT2Simple, GPT2SimpleConfig
from spd.utils.module_utils import expand_module_patterns

DEVICE = "cpu"


@pytest.fixture
def app_with_state():
    """Set up app state with a fresh randomly-initialized model.

    This fixture:
    1. Creates an in-memory SQLite database
    2. Creates a fake "run" in the database
    3. Creates a fresh GPT2Simple model (randomly initialized, 1 layer)
    4. Wraps it in a fresh ComponentModel (randomly initialized)
    5. Constructs a test Config with small data split and short sequences
    6. Computes sources_by_target mapping
    7. Creates RunState and sets it on StateManager (with hardcoded token_strings, no real tokenizer)
    8. Returns FastAPI's TestClient
    """
    # Reset StateManager singleton for clean test state
    StateManager.reset()

    # Patch DEVICE in all router modules to use CPU for tests
    with (
        mock.patch.object(graphs_router, "DEVICE", DEVICE),
        mock.patch.object(prompts_router, "DEVICE", DEVICE),
        mock.patch.object(runs_router, "DEVICE", DEVICE),
    ):
        db = PromptAttrDB(db_path=Path(":memory:"), check_same_thread=False)
        db.init_schema()

        run_id = db.create_run("wandb:test/test/runs/testrun1")
        run = db.get_run(run_id)
        assert run is not None

        model_config = GPT2SimpleConfig(
            model_type="GPT2Simple",
            block_size=16,
            vocab_size=4019,  # Match tokenizer
            n_layer=1,
            n_head=2,
            n_embd=32,
            flash_attention=False,
        )
        target_model = GPT2Simple(model_config)
        target_model.eval()
        target_model.requires_grad_(False)

        target_module_patterns = [
            "h.*.mlp.c_fc",
            "h.*.mlp.down_proj",
            "h.*.attn.q_proj",
            "h.*.attn.k_proj",
            "h.*.attn.v_proj",
            "h.*.attn.o_proj",
        ]
        C = 8

        config = Config(
            n_mask_samples=1,
            ci_fn_type="shared_mlp",
            ci_fn_hidden_dims=[16],
            sampling="continuous",
            sigmoid_type="leaky_hard",
            module_info=[
                ModulePatternInfoConfig(module_pattern=p, C=C) for p in target_module_patterns
            ],
            pretrained_model_class="spd.pretrain.models.gpt2_simple.GPT2Simple",
            pretrained_model_output_attr="idx_0",
            tokenizer_name="SimpleStories/test-SimpleStories-gpt2-1.25M",
            output_loss_type="kl",
            lr_schedule=ScheduleConfig(start_val=1e-3),
            steps=1,
            batch_size=1,
            eval_batch_size=1,
            n_eval_steps=1,
            eval_freq=1,
            slow_eval_freq=1,
            train_log_freq=1,
            task_config=LMTaskConfig(
                task_name="lm",
                max_seq_len=3,  # Short sequences
                dataset_name="SimpleStories/SimpleStories",
                column_name="story",
                train_data_split="test[:20]",  # Only 20 samples
                eval_data_split="test[:20]",
            ),
        )
        module_path_info = expand_module_patterns(target_model, config.module_info)
        model = ComponentModel(
            target_model=target_model,
            module_path_info=module_path_info,
            ci_fn_type=config.ci_fn_type,
            ci_fn_hidden_dims=config.ci_fn_hidden_dims,
            pretrained_model_output_attr=config.pretrained_model_output_attr,
            sigmoid_type=config.sigmoid_type,
        )
        model.eval()
        sources_by_target = get_sources_by_target(
            model=model, device=DEVICE, sampling=config.sampling
        )

        # The model has vocab_size=4019, so create entries for all token IDs
        token_strings = {i: f"tok_{i}" for i in range(model_config.vocab_size)}

        run_state = RunState(
            run=run,
            model=model,
            context_length=1,
            tokenizer=None,  # pyright: ignore[reportArgumentType]
            sources_by_target=sources_by_target,
            config=config,
            token_strings=token_strings,
            harvest=HarvestCache(run_id="test_run"),
        )

        manager = StateManager.get()
        manager.initialize(db)
        manager.run_state = run_state

        yield TestClient(app)

        manager.close()
        StateManager.reset()


@pytest.fixture
def app_with_prompt(app_with_state: TestClient) -> tuple[TestClient, int]:
    """Extends app_with_state with a pre-created prompt for graph tests.

    Returns:
        Tuple of (TestClient, prompt_id)
    """
    manager = StateManager.get()
    assert manager.run_state is not None
    prompt_id = manager.db.add_custom_prompt(
        run_id=manager.run_state.run.id,
        token_ids=[0, 2, 1],
        active_components={},  # Empty for testing
        context_length=manager.run_state.context_length,
    )
    return app_with_state, prompt_id


# -----------------------------------------------------------------------------
# Health Check
# -----------------------------------------------------------------------------


def test_health_check(app_with_state: TestClient):
    """Test that health endpoint returns ok status."""
    response = app_with_state.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# -----------------------------------------------------------------------------
# Run Management
# -----------------------------------------------------------------------------


def test_get_status(app_with_state: TestClient):
    """Test getting current server status with loaded run."""
    response = app_with_state.get("/api/status")
    assert response.status_code == 200
    status = response.json()
    assert status["wandb_path"] == "wandb:test/test/runs/testrun1"
    assert "config_yaml" in status


# -----------------------------------------------------------------------------
# Compute
# -----------------------------------------------------------------------------


def test_compute_graph(app_with_prompt: tuple[TestClient, int]):
    """Test computing attribution graph for a prompt."""
    client, prompt_id = app_with_prompt
    response = client.post(
        "/api/graphs",
        params={"prompt_id": prompt_id, "normalize": "none", "ci_threshold": 0.0},
    )
    assert response.status_code == 200

    # Parse SSE stream
    lines = response.text.strip().split("\n")
    events = [line for line in lines if line.startswith("data:")]
    assert len(events) >= 1

    # Final event should be complete with graph data
    final_data = json.loads(events[-1].replace("data: ", ""))
    assert final_data["type"] == "complete"

    data = final_data["data"]
    assert "edges" in data
    assert "tokens" in data
    assert "outputProbs" in data


# -----------------------------------------------------------------------------
# Streaming: Prompt Generation
# -----------------------------------------------------------------------------


@pytest.mark.slow
def test_generate_prompts_streaming(app_with_state: TestClient):
    """Test streaming prompt generation with CI harvesting."""
    response = app_with_state.post("/api/prompts/generate", params={"n_prompts": 2})
    assert response.status_code == 200

    # Parse SSE stream
    lines = response.text.strip().split("\n")
    events = [line for line in lines if line.startswith("data:")]

    # Should have progress events and a complete event
    assert len(events) >= 1

    # Final event should be complete
    final_data = json.loads(events[-1].replace("data: ", ""))
    assert final_data["type"] == "complete"


# -----------------------------------------------------------------------------
# Prompts and Search
# -----------------------------------------------------------------------------


@pytest.mark.slow
def test_get_prompts_initially_empty(app_with_state: TestClient):
    """Test that prompts list is initially empty."""
    response = app_with_state.get("/api/prompts")
    assert response.status_code == 200
    prompts = response.json()
    assert len(prompts) == 0


@pytest.mark.slow
def test_get_prompts_after_generation(app_with_state: TestClient):
    """Test getting prompts after generation."""
    # Generate some prompts first
    app_with_state.post("/api/prompts/generate", params={"n_prompts": 2})

    response = app_with_state.get("/api/prompts")
    assert response.status_code == 200
    prompts = response.json()
    assert len(prompts) >= 2


@pytest.mark.slow
def test_search_prompts(app_with_state: TestClient):
    """Test searching prompts by component keys."""
    # Generate prompts first
    app_with_state.post("/api/prompts/generate", params={"n_prompts": 2})

    # Search for a component that should exist (wte:0 is always active)
    response = app_with_state.get(
        "/api/prompts/search",
        params={"components": "wte:0", "mode": "any"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert data["count"] >= 0


# -----------------------------------------------------------------------------
# Activation Contexts
# -----------------------------------------------------------------------------


def test_activation_contexts_not_found_initially(app_with_state: TestClient):
    """Test that activation contexts return 404 when not generated."""
    response = app_with_state.get("/api/activation_contexts/summary")
    assert response.status_code == 404


# -----------------------------------------------------------------------------
# Optimized Compute (Streaming)
# -----------------------------------------------------------------------------


@pytest.mark.slow
def test_compute_optimized_stream(app_with_prompt: tuple[TestClient, int]):
    """Test streaming optimized attribution computation."""
    client, prompt_id = app_with_prompt
    response = client.post(
        "/api/graphs/optimized/stream",
        params={
            "prompt_id": prompt_id,
            "label_token": 2,
            "imp_min_coeff": 0.01,
            "ce_loss_coeff": 1.0,
            "steps": 5,  # Very few steps for testing
            "pnorm": 0.5,
            "beta": 0.5,
            "normalize": "none",
            "ci_threshold": 0.0,
            "output_prob_threshold": 0.01,
        },
    )
    assert response.status_code == 200

    events = [line for line in response.text.strip().split("\n") if line.startswith("data:")]
    assert len(events) >= 1
