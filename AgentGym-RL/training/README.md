# Slurm launchers

Each `<task>_train.sh` here is a Slurm job: it starts the task's environment server
(`start_env_server.sh`, ports in `port_map.txt`), waits until the server accepts connections,
then trains. `multiple_train.sh` runs several tasks in one job, one GPU each.

## One-time setup: account and partition

The scripts contain no account or partition. Slurm reads both from the environment, so each
user sets their own in `~/.bashrc`:

```sh
export SBATCH_ACCOUNT=<allocation> SBATCH_PARTITION=<partition>   # sbatch
export SALLOC_ACCOUNT=<allocation> SALLOC_PARTITION=<partition>   # salloc
export SLURM_ACCOUNT=<allocation>  SLURM_PARTITION=<partition>    # srun
```

On Quest, GPU jobs use the `gengpu` partition. Command-line flags override the environment
for a single job:

```sh
sbatch -A <allocation> -p <partition> AgentGym-RL/training/sciworld_train.sh
```

## Usage

Submit from the repository root:

```sh
sbatch AgentGym-RL/training/textworldexpress_train.sh
FLAVOR=ScalingInter-RL sbatch AgentGym-RL/training/textworldexpress_train.sh
sbatch AgentGym-RL/training/multiple_train.sh
```

Job logs are written to `output_<jobid>.log` and `error_<jobid>.log` in the submit directory.
