#!/bin/bash

# CamStream Setup Script
echo "🎥 Setting up CamStream..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This project is designed for macOS only"
    exit 1
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew is not installed. Please install it first:"
    echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "📦 Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
fi

# Install FFmpeg
echo "📦 Installing FFmpeg..."
brew install ffmpeg

# Create virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Build PyO3 module
echo "🔨 Building PyO3 module..."
maturin develop

# Build Rust binary
echo "🔨 Building Rust binary..."
cargo build --bin camstream

echo "✅ Setup complete!"
echo ""
echo "To run the camera analyzer:"
echo "  source .venv/bin/activate"
echo "  ./target/debug/camstream"
echo ""
echo "To test your setup:"
echo "  source .venv/bin/activate"
echo "  python test_setup.py"
