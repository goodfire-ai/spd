# SPD Visualization App

A lightweight web app for visualizing SPD decomposition results. This is a pure Svelte SPA with a FastAPI backend.

## Quick Start

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
npm run build        # Build for production
npm run check        # Type check (like mypy or basedpyright)
```

### Svelte

**What is it?** A reactive UI framework that compiles to vanilla JavaScript. Unlike React, there's no virtual DOM - it compiles to efficient imperative code.

**Basic .svelte file structure**:
```svelte
<script lang="ts">
    // TypeScript code - runs once when component is created
    let count = 0;

    // Reactive statements - re-run when dependencies change
    $: doubled = count * 2;
</script>

<!-- HTML markup with reactive bindings -->
<button on:click={() => count++}>
    Count: {count}, Doubled: {doubled}
</button>

<style>
    /* Scoped CSS - only applies to this component */
    button { color: blue; }
</style>
```

**Key Svelte features used in this app**:
- `let` variables are reactive - updating them triggers UI updates
- `$:` prefix creates reactive statements that automatically re-run
- `bind:value={variable}` creates two-way binding
- `on:click={handler}` attaches event handlers
- `{#if condition}...{/if}` for conditional rendering

## Architecture

### Data Flow (End-to-End Example)

Let's trace how loading a W&B run works:

1. **User Input** ([App.svelte:64-76](frontend/src/App.svelte#L64-L76))
   - User enters W&B run path in input field
   - Clicks "Load Run" button
   - `loadRun()` function is called

2. **Frontend API Call** ([api.ts:16-25](frontend/src/lib/api.ts#L16-L25))
   ```typescript
   export async function loadRun(wandbRunPath: string): Promise<void> {
       const response = await fetch(`${API_URL}/runs/load`, {
           method: "POST"
       });
   }
   ```

3. **Backend Endpoint** ([server.py:48-51](backend/server.py#L48-L51))
   ```python
   @app.post("/runs/load")
   def load_run(wandb_run_path: str):
       run_context_service.load_run(unquote(wandb_run_path))
   ```

4. **Service Layer** ([run_context_service.py:44-88](backend/services/run_context_service.py#L44-L88))
   - Downloads ComponentModel from W&B
   - Loads model onto GPU
   - Creates data loader with tokenizer
   - Stores in `RunContextService` singleton

5. **State Update** ([App.svelte:16-30](frontend/src/App.svelte#L16-L30))
   - Frontend polls `/status` endpoint every 5s
   - Receives updated status with loaded run info
   - Reactive statement triggers UI update
   ```typescript
   let status: Status = { train_run: null };
   // When status changes, UI automatically re-renders
   ```

6. **UI Renders** ([App.svelte:89-94](frontend/src/App.svelte#L89-L94))
   - Sidebar shows config YAML
   - "Activation Contexts" tab becomes available

### Another Example: Loading Activation Contexts

1. **User clicks "Load Contexts"** ([ActivationContextsTab.svelte:107-109](frontend/src/lib/components/ActivationContextsTab.svelte#L107-L109))

2. **API call with query parameters** ([api.ts:60-81](frontend/src/lib/api.ts#L60-L81))
   ```typescript
   const url = new URL(`${API_URL}/activation_contexts/subcomponents`);
   url.searchParams.set('importance_threshold', '0.0');
   url.searchParams.set('n_batches', '1');
   // ... more params
   const response = await fetch(url);
   ```

3. **Backend computation** ([server.py:54-71](backend/server.py#L54-L71))
   ```python
   @app.get("/activation_contexts/subcomponents")
   def get_subcomponent_activation_contexts(
       importance_threshold: float,
       n_batches: int,
       # ... more params
   ) -> ModelActivationContexts:
       return get_subcomponents_activation_contexts(...)
   ```

4. **Data flows back** through type-safe schemas:
   - Python Pydantic models ([schemas.py](backend/schemas.py))
   - TypeScript types ([api.ts:27-50](frontend/src/lib/api.ts#L27-L50))
   - These are manually kept in sync

5. **Component renders** ([ActivationContextsViewer.svelte](frontend/src/lib/components/ActivationContextsViewer.svelte))
   - Receives data as prop
   - Reactive statements compute derived values
   - UI updates automatically

### State Management

This app uses **simple, local state** - no Redux/Vuex/Pinia equivalent.

**Component-local state**:
```typescript
// Declaring a variable in <script> makes it reactive
let loading = false;
let data = null;

// Updating it triggers re-render
loading = true;  // UI automatically updates
```

**Derived state** (like `useMemo` in React):
```typescript
// Runs whenever `data` changes
$: processedData = data?.map(item => transform(item));
```

**Passing data between components**:
```svelte
<!-- Parent component -->
<script>
    let myData = { ... };
</script>
<ChildComponent data={myData} />

<!-- Child component -->
<script>
    export let data;  // "export" marks it as a prop
</script>
```

**Why no central store?** The app is simple enough that component-local state + props works fine. The backend is the source of truth - we just fetch what we need when we need it.

**Polling for updates**:
```typescript
onMount(() => {
    loadStatus();
    setInterval(loadStatus, 5000);  // Poll every 5s
});
```

## Project Structure

```
app/
├── backend/
│   ├── server.py              # FastAPI routes
│   ├── schemas.py             # Pydantic models (API contracts)
│   ├── services/              # Business logic
│   │   └── run_context_service.py
│   └── lib/                   # Utilities
│       └── activation_contexts.py
│
└── frontend/
    ├── package.json           # Dependencies & scripts
    ├── vite.config.ts         # Build tool config
    ├── src/
    │   ├── App.svelte         # Main app component
    │   ├── main.ts            # Entry point
    │   └── lib/
    │       ├── api.ts         # Backend API client
    │       ├── index.ts       # Utility functions
    │       └── components/    # Svelte components
    │           ├── ActivationContextsTab.svelte
    │           ├── ActivationContextsViewer.svelte
    │           ├── ActivationContext.svelte
    │           └── TokenHighlights.svelte
```

## Type Safety

Both frontend and backend use TypeScript/Python type annotations. However, the interface between them (API schemas) must be manually kept in sync:

- Backend: [schemas.py](backend/schemas.py) (Pydantic models)
- Frontend: [api.ts](frontend/src/lib/api.ts) (TypeScript types)

When you change the API, update both files.

## Development Tips

**Hot reload**: Both frontend (Vite) and backend (uvicorn) auto-reload on file changes.

**CORS**: Fully permissive in development since we're only running locally.

**Environment variables**:
- Frontend: Create `app/frontend/.env` with `VITE_API_URL=http://localhost:8000`
- Vite automatically loads `.env` files and exposes `VITE_*` variables

**Debugging**:
- Frontend: Browser DevTools Console (check Network tab for API calls)
- Backend: Logs print to terminal, tracebacks on errors
