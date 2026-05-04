# RL Pilot Data/Code Completion Report

## Requirements

- Requirement 1, framework running: completed.
  - BabyAI smoke: `7094697`
  - BabyAI benchmark: `7099097`
  - TextWorldExpress smoke: `7103978`, improved smoke: `7107157`
- Requirement 2, training curves: completed.
  - `babyai_smoke_training_curves.png`
  - `babyai_benchmark_training_curves.png`
  - `textworldexpress_smoke_training_curves.png`
  - `textworldexpress_benchmark_training_curves.png`
  - `smoke_comparison_training_curves.png`
  - CSV files under this directory.
- Requirement 3, case studies: completed.
  - `case_studies.md`

## Bonus Coverage

- More environments/baselines: BabyAI plus TextWorldExpress.
- New environment added: `AgentGym/agentenv-textworldexpress`.
- Baseline improvement: TextWorldExpress valid-action canonicalization.
- Finding: TextWorldExpress action grounding remains weak.

## Key Numbers

- BabyAI benchmark steps parsed: 302
- TextWorldExpress benchmark steps parsed: 319
- TextWorldExpress improved smoke steps parsed: 3
- TextWorldExpress `Unknown action`: 54 before canonicalization, 47 after canonicalization.

## Important Files

- `docs/BABYAI_RL_TRAINING_NOTES.md`
- `docs/TEXTWORLDEXPRESS_INTEGRATION_GUIDE.md`
- `slurm/babyai_rl_benchmark.sbatch`
- `slurm/textworldexpress_rl_smoke.sbatch`
- `scripts/build_pilot_artifacts.py`
