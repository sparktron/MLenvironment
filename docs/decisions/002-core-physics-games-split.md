# ADR-002: Separate core, physics, and games

The framework separates task-independent training, configuration, and utilities
in `core` from the walker in `physics` and the arena in `games`. A task registers
a `TaskSpec` and contributes its own environment and optional hooks.

`core` imports only `core`, with one explicit exception: `core.registry` imports
`PLUGIN_MODULES` from `rl_framework.plugins`. `physics` may import `core` and
`physics`; `games` may import `core` and `games`. Core may not contain task
names or task package paths as string literals. The repository policy checker
enforces these rules, including function-local and relative imports.

The CLI and GUI are application layers: they compose the framework and may
import any task package. Their public entry points and config paths remain in
place.

Built-in plugins are module names in `plugins.py`, loaded lazily by the registry.
The `RL_FRAMEWORK_PLUGINS` environment variable adds external modules. This
works with fresh subprocess workers and the project's lockfile installation and
root package path shim, without relying on installed setuptools entry points.
