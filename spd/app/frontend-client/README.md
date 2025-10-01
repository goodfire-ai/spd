# SPD Component Explorer - Client-Side App

This is a client-side only Svelte application (migrated from SvelteKit) for exploring and analyzing SPD (Stochastic Parameter Decomposition) components.

## Key Differences from SvelteKit Version

- **No server-side rendering**: Pure client-side application using Vite + Svelte
- **Direct API calls**: All API calls go directly to the backend at `http://localhost:8000`
- **Simpler routing**: No file-based routing - single-page application
- **Faster dev startup**: No SSR overhead

## Development

```bash
# Install dependencies
npm install

# Start dev server (with API proxy)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Type checking
npm run check

# Format code
npm run format

# Lint
npm run lint
```

## API Proxy

The Vite dev server is configured to proxy `/api` requests to `http://localhost:8000`. Make sure the backend is running on port 8000.

## Project Structure

```
src/
├── lib/
│   ├── api.ts              # API client
│   ├── components/         # Svelte components
│   └── stores/            # Svelte stores
├── App.svelte             # Main application component
├── main.ts                # Application entry point
└── app.css               # Global styles (with Tailwind)
```

## Notes

- All components and functionality from the SvelteKit version have been preserved
- The `$lib` alias is configured in `vite.config.ts` for consistency
- Tailwind CSS 4.0 is included for styling
