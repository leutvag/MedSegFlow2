#!/bin/bash

# Name of virtual environment
VENV_NAME=".venv"

echo "🔧 Creating virtual environment in $VENV_NAME..."
python3 -m venv $VENV_NAME

echo "⏳ Activating virtual environment..."
source $VENV_NAME/bin/activate

echo "⬆️  Upgrading pip..."
pip install --upgrade pip

echo "📦 Installing requirements..."
pip install -r requirements.txt

echo "✅ Setup complete. Virtual environment activated."
