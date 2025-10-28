# SPD Visualization App

A lightweight web app for visualizing SPD decomposition results. Built with Svelte 5 and FastAPI.

## Quick Start

**Option 1: All-in-one launcher (recommended)**

```bash
uv run app/run_app.py
```

This automatically starts both backend and frontend with health checks and port detection.

**Option 2: Manual startup**

```bash
# Terminal 1: Start backend
uv run app/run_backend.py

# Terminal 2: Start frontend
cd app/frontend
npm install
npm run dev
```

Then open http://localhost:5173 in your browser.

## For ML Engineers/Researchers: Web Dev Basics

### JavaScript/Node.js Ecosystem

**package.json**: The Python equivalent of `pyproject.toml` or `requirements.txt`. Defines:

- Dependencies (like `numpy`, `torch` in Python)
- Scripts (like `make` commands)
- Metadata about the project

**npm** (Node Package Manager): Like `pip` or `uv` for Python packages.

**Common commands**:

```bash
npm install          # Install dependencies (like pip install -r requirements.txt)
npm run dev          # Start development server
npm run check        # Type check (like mypy or basedpyright)
```

### Svelte 5

**Key Svelte 5 features used in this app**:

- `$state(value)` - reactive state (replaces `let` variables)
- `$derived(expression)` - computed values (replaces `$:` statements)
- `$effect(() => {})` - side effects (replaces `onMount`, `afterUpdate`)
- `bind:value={variable}` - two-way binding
- `onclick={handler}` - event handlers (replaces `on:click`)
- `{#if condition}...{/if}` - conditional rendering

## Architecture

### Data Flow (End-to-End Example)

Let's trace how loading a W&B run works:

1. **User Input** ([App.svelte:61-76](frontend/src/App.svelte#L61-L76))

   - User enters W&B run path in input field
   - Clicks "Load Run" button
   - `loadRun()` function is called

2. **Frontend API Call** ([api.ts](frontend/src/lib/api.ts))

   ```typescript
   export async function loadRun(wandbRunPath: string): Promise<void> {
     const response = await fetch(`${API_URL}/runs/load`, {
       method: "POST",
       body: JSON.stringify({ wandb_run_path: wandbRunPath }),
     });
   }
   ```

3. **Backend Endpoint** ([server.py](backend/server.py))

   ```python
   @app.post("/runs/load")
   def load_run(wandb_run_path: str):
       run_context_service.load_run(unquote(wandb_run_path))
   ```

4. **Service Layer** ([run_context_service.py](backend/services/run_context_service.py))

   - Downloads ComponentModel from W&B
   - Loads model onto GPU
   - Creates data loader with tokenizer
   - Stores in `RunContextService` singleton

5. **State Update** ([App.svelte:48-52](frontend/src/App.svelte#L48-L52))

   - Frontend polls `/status` endpoint every 5s using `$effect` rune
   - Receives updated status with loaded run info
   - Reactive state update triggers UI re-render

   ```typescript
   let status = $state<Status>({ train_run: null });

   $effect(() => {
     loadStatus();
     const interval = setInterval(loadStatus, 5000);
     return () => clearInterval(interval);
   });
   ```

6. **UI Renders** ([App.svelte:79-94](frontend/src/App.svelte#L79-L94))
   - Sidebar shows config YAML
   - "Activation Contexts" tab becomes available

## Project Structure

```
app/
├── run_app.py                 # All-in-one launcher
├── run_backend.py             # Backend-only launcher
├── backend/
│   ├── server.py              # FastAPI routes
│   ├── schemas.py             # Pydantic models (API contracts)
│   ├── services/              # Business logic
│   └── lib/                   # Utilities
└── frontend/
    ├── package.json           # Dependencies & scripts
    ├── vite.config.ts         # Build tool config (minimal Vite setup)
    ├── svelte.config.js       # Svelte compiler config
    ├── index.html             # SPA entry point
    └── src/
        ├── main.ts            # TypeScript entry point
        ├── App.svelte         # Root component
        └── lib/
            ├── api.ts         # Backend API client
            └── index.ts       # Utility functions
        └── components/    # Svelte components
```

## Type Safety

Both frontend and backend use TypeScript/Python type annotations. However, the interface between them (API schemas) must be manually kept in sync:

- Backend: [schemas.py](backend/schemas.py) (Pydantic models)
- Frontend: [api.ts](frontend/src/lib/api.ts) (TypeScript types)

When you change the API, update both files.
