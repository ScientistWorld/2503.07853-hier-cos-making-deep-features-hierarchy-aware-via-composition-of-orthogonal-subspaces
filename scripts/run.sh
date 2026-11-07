#!/bin/bash
# Hardening job (iteration 11): retrains the reference and CE baseline on the
# hardened protocol, trains better/weaker variants, freezes the baseline into
# the evaluators, scores the reference, and replays variants and shortcuts
# through the real evaluators. See redteam/hardening/run_hardening.sh.
set -euo pipefail
cd /home/user
bash redteam/hardening/run_hardening.sh
