#!/bin/bash
# Setup script for RL Agent 1
# This script sets up the virtual environment and installs all dependencies

set -e

echo "🚀 Setting up RL Agent 1..."
echo ""

# Check for Python 3.8+
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "✓ Found Python ${PYTHON_VERSION}"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf .venv
fi
python3 -m venv .venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip and install build tools
echo ""
echo "⬆️  Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# Install pybullet with workaround for macOS zlib issue (if on macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    echo "🔨 Installing pybullet (with macOS zlib workaround)..."
    CFLAGS="-Dfdopen=fdopen" CPPFLAGS="-Dfdopen=fdopen" pip install pybullet
else
    echo ""
    echo "🔨 Installing pybullet..."
    pip install pybullet
fi

# Install remaining dependencies
echo ""
echo "📚 Installing remaining dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "   source .venv/bin/activate"
echo ""
echo "Then start training with:"
echo "   cd src && python train.py"
echo ""

