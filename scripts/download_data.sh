#!/usr/bin/env bash
# Download the EVELINK data and models (about 35.6 GB) into data/.
# Run this on the GPU machine. The host must be reachable from that machine.
set -euo pipefail
URL="https://cogcomp.seas.upenn.edu/Data/event_linking/data_and_models.zip"
mkdir -p data
cd data
wget -c "$URL"
unzip -n data_and_models.zip
