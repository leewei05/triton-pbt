## Quickstart

```sh
# allocate GPU
sbatch ./scripts/tunnel.slr

# login to the allocated node
# activate conda env
source ./scripts/setup.sh
python tests/test_maximum.py

🚀 Starting Hypothesis tests for tl.maximum...
✅ All randomized tests passed!
```