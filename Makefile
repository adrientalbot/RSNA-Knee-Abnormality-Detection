.PHONY: help setup sync notebook kernel clean

UV ?= uv
UV_CACHE_DIR ?= .uv-cache
KERNEL_NAME ?= rsna-knee

export UV_CACHE_DIR

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n\nTargets:\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Create .venv and install locked dependencies
	$(UV) sync

sync: ## Update .venv from uv.lock
	$(UV) sync --frozen

notebook: ## Start JupyterLab inside the project environment
	$(UV) run jupyter lab notebooks

kernel: ## Register the project environment as a Jupyter kernel
	$(UV) run python -m ipykernel install --user --name $(KERNEL_NAME) --display-name "Python (RSNA Knee)"

clean: ## Remove generated local caches (keeps .venv)
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .ipynb_checkpoints \) -prune -exec rm -rf {} +
