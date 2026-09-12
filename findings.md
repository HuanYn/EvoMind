# Research findings

## 2026-09-12 — Text base selection for Dense single-image initialization

Independent result-to-claim review: `claim_supported=yes` for the narrow engineering selection claim; confidence `medium`. Accept final CISPO checkpoint `171d23ad98338540617532ca101c47045210203c798547e781141dcb77d0f1ed` from `artifacts/runs/cispo_gpu2_product_20260909/full/weights/cispo_768.pth`. Preserve the completed same-DPO GRPO branch as comparison evidence.

Both final RL checkpoints pass the configured DPO-relative thresholds: all seven native `acc,none` decreases are within 0.01, EOS is unchanged, and thinking-off repeat-4 decreases by 0.02370578268615518 for CISPO and 0.02369253165718188 for GRPO. The product route selects CISPO; no seven-task average or benchmark-win tie-breaker is defined. GRPO would also fail the strict zero-repeat-increase rule as a subsequent CISPO replacement by +0.00001325102897330, which is not evidence of meaningful CISPO quality superiority.

Selection uses the eight thinking-off diagnostics. Thinking-on results remain separate. C-Eval/CMMLU use native sample-weighted group accuracy across 1,346/11,582 rows, not unweighted subject means; `acc_norm,none` is not substituted. Both formal RL runs completed one epoch and 9,751 optimizer updates from the same DPO checkpoint and full 19,502-row dataset.

The result does not establish statistical superiority, visual transfer, video ability or agent success. There is one training seed; installed harness source hashes differ between local SFT/DPO and remote CISPO/GRPO despite matching evaluator/model/tokenizer hashes and package versions. ToolCall completion is not correctness: both RL models answer that 35 squared is 35, and both finish the Tokyo weather/conversion case without a tool call. Agent-CISPO remains deferred and nonblocking.

At review time, all 28 archived summary hashes, selected scores and declared raw-result hashes matched the public snapshot. The reviewer did not independently verify absent local copies of RL harness `results.json`, thinking `records.jsonl` or remote checkpoint bytes. The local product state was still `remote_training`, without a final product acceptance receipt. The executor must finish strict raw/checkpoint import, preserve cross-environment provenance, reconcile the completed state and bind the acceptance receipt before the executable vision gate opens. Root reports that this import is now in progress; this entry does not certify its completion.

Routing: confirm the narrow engineering claim, finish artifact acceptance, then continue the already-authorized Dense single-image stage and its acceptance checks. No new SFT/DPO benchmark run or supplementary text training is required. No `EXPERIMENT_AUDIT.json` was found, so the review carries the nonblocking label “provisional — no integrity audit run.”

Full local trace: `.aris/traces/result-to-claim/2026-09-12_run01/`. The trace contains the exact review request, full response, metadata and parsed verdict/routing; keep `.aris/traces/` out of commits.

### Executor acceptance completed

The executor subsequently retrieved the original RL `results.json`, `records.jsonl` and both final checkpoints, verified all 28 native benchmark raw files against the pinned published hashes and full coverage, and recomputed the selection. The accepted checkpoint remains CISPO `171d23ad…`. The original remote evaluation loader is preserved byte-for-byte because the later public loader added training-checkpoint wrapper support; this source difference is not erased from historical evidence.

`product_acceptance.json` now binds the import spec, original evidence and selected checkpoint; the vision scope is enabled locally only after re-verification. Twenty new import tests and twelve existing product tests pass. This closes the text initialization gate, not the visual-quality gate. See [selection and handoff](docs/TEXT_BASE_SELECTION.md).
