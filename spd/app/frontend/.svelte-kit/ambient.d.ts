
// this file is generated — do not edit it


/// <reference types="@sveltejs/kit" />

/**
 * Environment variables [loaded by Vite](https://vitejs.dev/guide/env-and-mode.html#env-files) from `.env` files and `process.env`. Like [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), this module cannot be imported into client-side code. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured).
 * 
 * _Unlike_ [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), the values exported from this module are statically injected into your bundle at build time, enabling optimisations like dead code elimination.
 * 
 * ```ts
 * import { API_KEY } from '$env/static/private';
 * ```
 * 
 * Note that all environment variables referenced in your code should be declared (for example in an `.env` file), even if they don't have a value until the app is deployed:
 * 
 * ```
 * MY_FEATURE_FLAG=""
 * ```
 * 
 * You can override `.env` values from the command line like so:
 * 
 * ```sh
 * MY_FEATURE_FLAG="enabled" npm run dev
 * ```
 */
declare module '$env/static/private' {
	export const OLD_CPATH: string;
	export const LESSOPEN: string;
	export const LIBRARY_PATH: string;
	export const NCCL_IB_PCI_RELAXED_ORDERING: string;
	export const NCCL_SOCKET_IFNAME: string;
	export const NCCL_IB_HCA: string;
	export const USER: string;
	export const SSH_CLIENT: string;
	export const npm_config_user_agent: string;
	export const HPCX_OSU_CUDA_DIR: string;
	export const GIT_ASKPASS: string;
	export const npm_node_execpath: string;
	export const SHLVL: string;
	export const LD_LIBRARY_PATH: string;
	export const BROWSER: string;
	export const npm_config_noproxy: string;
	export const MOTD_SHOWN: string;
	export const HOME: string;
	export const HPCX_UCX_DIR: string;
	export const OLDPWD: string;
	export const TERM_PROGRAM_VERSION: string;
	export const HF_DATASETS_CACHE: string;
	export const OSHMEM_HOME: string;
	export const VSCODE_IPC_HOOK_CLI: string;
	export const npm_package_json: string;
	export const VSCODE_GIT_ASKPASS_MAIN: string;
	export const MODULES_CMD: string;
	export const PS1: string;
	export const VSCODE_GIT_ASKPASS_NODE: string;
	export const npm_config_userconfig: string;
	export const npm_config_local_prefix: string;
	export const OLD_PATH: string;
	export const PYDEVD_DISABLE_FILE_VALIDATION: string;
	export const BUNDLED_DEBUGPY_PATH: string;
	export const SPD_CACHE_DIR: string;
	export const npm_config_engine_strict: string;
	export const COLORTERM: string;
	export const COLOR: string;
	export const HPCX_OSU_DIR: string;
	export const OMPI_HOME: string;
	export const LOGNAME: string;
	export const OLD_LIBRARY_PATH: string;
	export const OPAL_PREFIX: string;
	export const _: string;
	export const npm_config_prefix: string;
	export const npm_config_npm_version: string;
	export const PKG_CONFIG_PATH: string;
	export const TERM: string;
	export const npm_config_cache: string;
	export const HPCX_HCOLL_DIR: string;
	export const HF_HUB_CACHE: string;
	export const SLURM_JOB_ID: string;
	export const OLD_LD_LIBRARY_PATH: string;
	export const npm_config_node_gyp: string;
	export const PATH: string;
	export const MPI_HOME: string;
	export const NODE: string;
	export const npm_package_name: string;
	export const HPCX_DIR: string;
	export const VSCODE_DEBUGPY_ADAPTER_ENDPOINTS: string;
	export const HPCX_UCC_DIR: string;
	export const LANG: string;
	export const VIRTUAL_ENV_PROMPT: string;
	export const LS_COLORS: string;
	export const VSCODE_GIT_IPC_HANDLE: string;
	export const TERM_PROGRAM: string;
	export const npm_lifecycle_script: string;
	export const SSH_AUTH_SOCK: string;
	export const HPCX_MPI_TESTS_DIR: string;
	export const SHELL: string;
	export const npm_package_version: string;
	export const npm_lifecycle_event: string;
	export const HPCX_OSHMEM_DIR: string;
	export const LESSCLOSE: string;
	export const HPCX_SHARP_DIR: string;
	export const VIRTUAL_ENV: string;
	export const VSCODE_GIT_ASKPASS_EXTRA_ARGS: string;
	export const npm_config_globalconfig: string;
	export const npm_config_init_module: string;
	export const JAVA_HOME: string;
	export const PWD: string;
	export const SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING: string;
	export const LOADEDMODULES: string;
	export const HPCX_NCCL_RDMA_SHARP_PLUGIN_DIR: string;
	export const npm_execpath: string;
	export const SSH_CONNECTION: string;
	export const UCX_VFS_ENABLE: string;
	export const npm_config_global_prefix: string;
	export const HPCX_MPI_DIR: string;
	export const HPCX_IPM_DIR: string;
	export const npm_command: string;
	export const NVIDIA_VISIBLE_DEVICES: string;
	export const CPATH: string;
	export const HPCX_CLUSTERKIT_DIR: string;
	export const MANPATH: string;
	export const MODULEPATH: string;
	export const MODULESHOME: string;
	export const INIT_CWD: string;
	export const EDITOR: string;
	export const NODE_ENV: string;
}

/**
 * Similar to [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private), except that it only includes environment variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Values are replaced statically at build time.
 * 
 * ```ts
 * import { PUBLIC_BASE_URL } from '$env/static/public';
 * ```
 */
declare module '$env/static/public' {
	
}

/**
 * This module provides access to runtime environment variables, as defined by the platform you're running on. For example if you're using [`adapter-node`](https://github.com/sveltejs/kit/tree/main/packages/adapter-node) (or running [`vite preview`](https://svelte.dev/docs/kit/cli)), this is equivalent to `process.env`. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured).
 * 
 * This module cannot be imported into client-side code.
 * 
 * ```ts
 * import { env } from '$env/dynamic/private';
 * console.log(env.DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 * 
 * > [!NOTE] In `dev`, `$env/dynamic` always includes environment variables from `.env`. In `prod`, this behavior will depend on your adapter.
 */
declare module '$env/dynamic/private' {
	export const env: {
		OLD_CPATH: string;
		LESSOPEN: string;
		LIBRARY_PATH: string;
		NCCL_IB_PCI_RELAXED_ORDERING: string;
		NCCL_SOCKET_IFNAME: string;
		NCCL_IB_HCA: string;
		USER: string;
		SSH_CLIENT: string;
		npm_config_user_agent: string;
		HPCX_OSU_CUDA_DIR: string;
		GIT_ASKPASS: string;
		npm_node_execpath: string;
		SHLVL: string;
		LD_LIBRARY_PATH: string;
		BROWSER: string;
		npm_config_noproxy: string;
		MOTD_SHOWN: string;
		HOME: string;
		HPCX_UCX_DIR: string;
		OLDPWD: string;
		TERM_PROGRAM_VERSION: string;
		HF_DATASETS_CACHE: string;
		OSHMEM_HOME: string;
		VSCODE_IPC_HOOK_CLI: string;
		npm_package_json: string;
		VSCODE_GIT_ASKPASS_MAIN: string;
		MODULES_CMD: string;
		PS1: string;
		VSCODE_GIT_ASKPASS_NODE: string;
		npm_config_userconfig: string;
		npm_config_local_prefix: string;
		OLD_PATH: string;
		PYDEVD_DISABLE_FILE_VALIDATION: string;
		BUNDLED_DEBUGPY_PATH: string;
		SPD_CACHE_DIR: string;
		npm_config_engine_strict: string;
		COLORTERM: string;
		COLOR: string;
		HPCX_OSU_DIR: string;
		OMPI_HOME: string;
		LOGNAME: string;
		OLD_LIBRARY_PATH: string;
		OPAL_PREFIX: string;
		_: string;
		npm_config_prefix: string;
		npm_config_npm_version: string;
		PKG_CONFIG_PATH: string;
		TERM: string;
		npm_config_cache: string;
		HPCX_HCOLL_DIR: string;
		HF_HUB_CACHE: string;
		SLURM_JOB_ID: string;
		OLD_LD_LIBRARY_PATH: string;
		npm_config_node_gyp: string;
		PATH: string;
		MPI_HOME: string;
		NODE: string;
		npm_package_name: string;
		HPCX_DIR: string;
		VSCODE_DEBUGPY_ADAPTER_ENDPOINTS: string;
		HPCX_UCC_DIR: string;
		LANG: string;
		VIRTUAL_ENV_PROMPT: string;
		LS_COLORS: string;
		VSCODE_GIT_IPC_HANDLE: string;
		TERM_PROGRAM: string;
		npm_lifecycle_script: string;
		SSH_AUTH_SOCK: string;
		HPCX_MPI_TESTS_DIR: string;
		SHELL: string;
		npm_package_version: string;
		npm_lifecycle_event: string;
		HPCX_OSHMEM_DIR: string;
		LESSCLOSE: string;
		HPCX_SHARP_DIR: string;
		VIRTUAL_ENV: string;
		VSCODE_GIT_ASKPASS_EXTRA_ARGS: string;
		npm_config_globalconfig: string;
		npm_config_init_module: string;
		JAVA_HOME: string;
		PWD: string;
		SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING: string;
		LOADEDMODULES: string;
		HPCX_NCCL_RDMA_SHARP_PLUGIN_DIR: string;
		npm_execpath: string;
		SSH_CONNECTION: string;
		UCX_VFS_ENABLE: string;
		npm_config_global_prefix: string;
		HPCX_MPI_DIR: string;
		HPCX_IPM_DIR: string;
		npm_command: string;
		NVIDIA_VISIBLE_DEVICES: string;
		CPATH: string;
		HPCX_CLUSTERKIT_DIR: string;
		MANPATH: string;
		MODULEPATH: string;
		MODULESHOME: string;
		INIT_CWD: string;
		EDITOR: string;
		NODE_ENV: string;
		[key: `PUBLIC_${string}`]: undefined;
		[key: `${string}`]: string | undefined;
	}
}

/**
 * Similar to [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), but only includes variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Note that public dynamic environment variables must all be sent from the server to the client, causing larger network requests — when possible, use `$env/static/public` instead.
 * 
 * ```ts
 * import { env } from '$env/dynamic/public';
 * console.log(env.PUBLIC_DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 */
declare module '$env/dynamic/public' {
	export const env: {
		[key: `PUBLIC_${string}`]: string | undefined;
	}
}
