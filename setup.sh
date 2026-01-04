#!/bin/bash
# Setup script for RL Agent 1
# This script sets up the virtual environment and installs all dependencies

set -e  # Exit on error

echo "🚀 Setting up RL Agent 1..."

# Check for Python 3.11
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Error: python3.11 not found. Please install Python 3.11."
    echo "   You can install it via Homebrew: brew install python@3.11"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3.11 -m venv .venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip and install build tools
echo "⬆️  Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel cmake

# Install pybullet with workaround for macOS zlib issue
echo "🔨 Installing pybullet (with macOS zlib workaround)..."
CFLAGS="-Dfdopen=fdopen" CPPFLAGS="-Dfdopen=fdopen" pip install pybullet

# Install remaining dependencies
echo "📚 Installing remaining dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete! Activate the virtual environment with:"
echo "   source .venv/bin/activate"

