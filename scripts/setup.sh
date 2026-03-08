#!/bin/bash

# This script sets up the environment for the convergent-minds project.
# It creates a virtual environment, installs dependencies, and downloads necessary data.

set -e

echo "--- Setting up your environment for Convergent Minds ---"
echo

# 1. Create Python virtual environment
echo "[1/5] Creating Python virtual environment in ./.venv..."
python3 -m venv .venv
echo "Done."
echo

# 2. Activate virtual environment
# Note: activation is only for this script's session.
# User will need to run `source .venv/bin/activate` manually for future sessions.
source .venv/bin/activate

# 3. Install Python dependencies
echo "[2/5] Installing core Python dependencies..."
pip install torch transformers numpy pandas scipy tqdm xarray
echo "Done."
echo

# 4. Install brain-score-language package
echo "[3/5] Installing brain-score-language package in editable mode..."
pip install -e language
echo "Done."
echo

# 5. Download and extract data
echo "[4/5] Downloading and extracting Fedorenko2010 stimuli..."
echo "Note: This requires wget and unzip to be installed."
echo "The download URL is from Dropbox and may require access."

if [ ! -f fedorenko10_stimuli.zip ]; then
    # Using dl=1 to force download from Dropbox link
    wget "https://www.dropbox.com/sh/c9jhmsy4l9ly2xx/AACQ41zipSZFj9mFbDfJJ9c4a?e=1&dl=1" -O fedorenko10_stimuli.zip
else
    echo "stimuli zip file (fedorenko10_stimuli.zip) already exists, skipping download."
fi
echo "Done."
echo

echo "[5/5] Unzipping stimuli..."
# The destination directory is based on the README.md
mkdir -p data/fedorenko10_stimuli
unzip -o fedorenko10_stimuli.zip -d data/fedorenko10_stimuli
echo "Done."
echo

echo "--- Setup complete! ---"
echo
echo "To activate the virtual environment in your shell, run:"
echo "source .venv/bin/activate"
