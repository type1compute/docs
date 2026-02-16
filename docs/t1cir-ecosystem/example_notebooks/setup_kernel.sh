#!/bin/bash
# Setup T1C tutorial environment and Jupyter kernel

set -e

echo "T1C Tutorials - Jupyter Kernel Setup"
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
python -m ipykernel install --user --name=t1cir-tutorials --display-name="T1C Tutorials"

echo ""
echo "Setup complete!"
echo ""
echo "To start Jupyter:"
echo "  source .venv/bin/activate"
echo "  jupyter notebook"
echo ""
echo "Then select 'T1C Tutorials' as the kernel in your notebook."
echo ""
