# Historical evaluation results

These are the recorded simulation results behind the repository's README table
and comparison figure. They are not the output of the current workstation smoke
test, model weights, TensorBoard histories, or a complete training configuration.

| Files | Contents |
|---|---|
| `summary.json` | Baseline and expanded-run summaries, selected checkpoint hashes and experiment context |
| `paired-results.csv` | Tabular comparison of the recorded experiments |
| `{g1,a3,x2_ultra}-lafan.json` | Per-clip evaluation and aggregate metrics for the LAFAN baseline policies |
| `{g1,a3,x2_ultra}-bones.json` | Expanded-policy evaluations, with separate `lafan` and `heldout` sections containing summaries and per-clip metrics |

Metrics include failure counts, normalized return, completion fraction and
full-clip-weighted return. Normalized return divides by executed steps, so an
early failure can still have a high score. Consult completion and failures too.

Current defaults live in `src/supermimic/config.py`; each new training invocation
writes its resolved `config.json`, events and checkpoints under `logs/<run>/`.
The evaluation CLI writes beside the checkpoint by default. Keep new results in
that run directory until choosing to publish them explicitly.

The historical runs used additional `anchor_obs`, selected checkpoints and some
earlier world-model warmstarts. Weights and full training logs are not released;
running the public training commands does not guarantee these exact scores.
See [experiment settings and limitations](../EXPERIMENTS.md).
