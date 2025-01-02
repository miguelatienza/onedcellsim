#!/bin/bash
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PATH_TO_JULIA=$script_dir/venv/julia-1.6.7/bin/julia

source $script_dir/venv/bin/activate
python /home/miguel/onedcellsim_draft/onedcellsim/onedcellsim/simulators/multistability/viewer.py ${PATH_TO_JULIA}