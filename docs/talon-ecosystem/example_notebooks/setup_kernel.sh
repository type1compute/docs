#!/bin/bash
# Setup TALON tutorial environment and Jupyter kernel

set -e

echo "TALON Tutorials - Jupyter Kernel Setup"
echo "====================================="
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this script from the example_notebooks directory"
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate and install
echo "Installing dependencies..."
source .venv/bin/activate
uv pip install -r requirements.txt

# Register kernel
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name=talon-tutorials --display-name="TALON Tutorials"

echo ""
echo "Setup complete!"
echo ""
echo "To start Jupyter:"
echo "  source .venv/bin/activate"
echo "  jupyter notebook"
echo ""
echo "Then select 'TALON Tutorials' as the kernel in your notebook."
echo ""
