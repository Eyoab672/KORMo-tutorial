set -e

env_dir=".venv_kormo"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing uv..."
  python3 -m pip install --upgrade pip
  python3 -m pip install uv
else
  echo "uv found: $(command -v uv)"
fi

echo "Creating uv venv: $env_dir"

uv venv "$env_dir" --python=3.12
source "$env_dir/bin/activate"

echo "Currently in env $(which python)"

uv pip install -e .
uv pip install flash-attn --no-build-isolation

echo "Environment \"$env_dir\" created and all packages installed successfully!"