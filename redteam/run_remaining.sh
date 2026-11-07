#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# The previous allocation orphaned before seed 9 finished. Its model was not
# yet saved, so rerun this seed fully at the original scale. Subsequent
# interruptions can resume exactly from the new periodic RNG/optimizer state.
bash redteam/variant_views2/run.sh
bash redteam/variant_views10/run.sh
bash redteam/reference_seed_9/run.sh
printf '0\n' > redteam/job_exit_status.txt
