"""Durable continuation with an explicit text-before-vision scope gate.

This entry point uses no GPU itself. Stages run serially through evomind_run.
--plan-only materializes the audited commands without starting any training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

from evomind_run import atomic_json, run_manifest, now

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/vision_pipeline.json"
TEXT_SCOPE = ROOT / "configs/text_alignment_scope.json"


def require_vision_enabled(scope_path=TEXT_SCOPE):
    """The active supervisor already loaded its old manifest; gate its future entry point.

    Do not modify or restart the active pretrain just to change a later stage.
    A disabled/missing route must never fall through to a GPU vision experiment.
    """
    if not scope_path.is_file():
        raise RuntimeError("Text/vision scope contract is missing; refusing to start vision")
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    if scope.get("vision_enabled") is not True:
        raise RuntimeError("VISION_HELD_FOR_FULL_TEXT_ALIGNMENT: finish MiniMind-3 post-training branches and text evaluation first; see configs/text_alignment_scope.json")


def build_manifest():
    steps = []

    def action(name, *args):
        steps.append({"name": name, "cwd": ".",
                      "argv": ["scripts/evomind_pipeline_step.py", *args]})

    action("verify_assets_and_transfer_base", "setup")
    action("prepare_image_disjoint_data", "prepare")
    action("official_memory_probe", "probe")
    action("real_encoder_cache_parity", "parity")
    action("prepare_frozen_feature_cache", "cache")
    for variant in "ABC":
        action(f"smoke_{variant}", "train", "--variant", variant, "--seed", "42", "--smoke")
    action("official_v_two_epochs", "official")
    action("official_v_test", "evaluate", "--variant", "official", "--seed", "42")
    for seed in (42, 123, 2026):
        for variant in "ABC":
            action(f"full_{variant}_seed{seed}", "train", "--variant", variant, "--seed", str(seed))
            action(f"test_{variant}_seed{seed}", "evaluate", "--variant", variant, "--seed", str(seed))
    action("aggregate_report", "finalize")
    return {"run_dir": "artifacts/runs/vision_pipeline_20260908", "description":
            "Official 2-epoch image-disjoint reference plus fixed-budget 20k A/B/C, three training seeds. "
            "Human review cancelled by user; upstream example evaluation only, no quality-win claim.",
            "stages": steps}


def wait_for_human_annotations(run_dir, evaluate, sleep=time.sleep):
    """Keep the existing local supervisor alive; never manufacture human scores.

    Poll only file metadata while unchanged. Revalidate evaluations on a changed
    annotation file; machine failures return instead of triggering hidden GPU reruns.
    This is not an app heartbeat or an autonomous error-repair service.
    """
    receipt_path = run_dir / "text_acceptance.json"
    fingerprint = None
    started = now()
    while True:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("machine_evaluations_complete") is not True:
            return 2
        human = receipt.get("human_review", {})
        if receipt.get("status") == "accepted" and human.get("status") == "annotated":
            atomic_json(run_dir / "human_wait.json", {"status": "accepted", "started_at": started, "updated_at": now()})
            return 0
        if human.get("status") not in ("awaiting_human_annotations", "failed"):
            return 2
        annotations = Path(receipt["evaluation_state"]).parent / "blind_review/annotations.csv"
        stat = annotations.stat() if annotations.exists() else None
        current = (stat.st_mtime_ns, stat.st_size) if stat else (None, None)
        if fingerprint is not None and current != fingerprint:
            fingerprint = current
            code = evaluate()
            if code not in (0, 2):
                return code
            continue
        if fingerprint is None:
            print(f"AWAITING_REAL_HUMAN_ANNOTATIONS: {annotations}; complete annotations to continue automatically. No synthetic human scores.", flush=True)
        fingerprint = current
        atomic_json(run_dir / "human_wait.json", {"status": "awaiting_real_human_annotations",
            "started_at": started, "updated_at": now(), "annotations": str(annotations),
            "poll_seconds": 30, "human_review": human})
        sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Explicit continuation after a failed/interrupted pipeline")
    parser.add_argument("--vision-only", action="store_true", help="Run vision only after the verified text acceptance gate")
    args = parser.parse_args()
    # Latest user-approved product route supersedes the historical branch queue.
    # The already running supervisor loads this file only after SFT finishes.
    if (ROOT / "configs/product_pipeline.json").is_file():
        from evomind_product import settings, run_product, enable_vision
        plan = settings()
        manifest = build_manifest()
        if not MANIFEST.exists() or json.loads(MANIFEST.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Vision manifest must match the preserved image-stage contract")
        print("Product route: SFT -> DPO -> selection -> CISPO -> selection -> image/video; Agent deferred", flush=True)
        if args.plan_only:
            print(f"{len(plan['stages'])} required text stages; {len(manifest['stages'])} existing image stages; video implementation pending", flush=True)
            return
        try:
            if not args.vision_only:
                run_product()
            enable_vision()
            require_vision_enabled()
            code = run_manifest(MANIFEST, resume=args.resume or (ROOT / manifest['run_dir'] / "state.json").exists())
            if code:
                raise SystemExit(code)
            atomic_json(ROOT / plan['run_dir'] / "next_stage.json", {
                "status": "video_implementation_pending", "image_pipeline_complete": True,
                "project_complete": False, "agent": "deferred_tool_extension_not_video_gate"})
        finally:
            import evomind_report
            evomind_report.main()
            try:
                import evomind_sync_notes
                evomind_sync_notes.main()
            except Exception as error:
                atomic_json(ROOT / plan['run_dir'] / "notes_sync_pending.json", {"error": str(error)})
        return
    # The active pretrain/SFT manifest already points to this future entry point.
    # Updating the entry point adds post-training without restarting live training.
    from evomind_posttrain import materialize, run_posttraining, RUN
    plan = materialize()
    manifest = build_manifest()
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing vision manifest differs; refusing to overwrite its experiment definition")
    else:
        atomic_json(MANIFEST, manifest)
    print(f"Full continuation: {len(plan['branches'])} post-SFT branches + text evaluation, then {len(manifest['stages'])} vision stages", flush=True)
    if not args.plan_only:
        if not args.vision_only:
            training_code = run_posttraining(resume=args.resume or (RUN / "state.json").exists())
            from evomind_posttrain_evaluate import run_evaluations
            evaluation_code = run_evaluations()  # evaluate successful branches even if another failed
            # Keep truthful reports and Obsidian in sync on both successful and incomplete execution.
            import evomind_report
            import evomind_sync_notes
            evomind_report.main()
            try:
                evomind_sync_notes.main()
            except Exception as error:
                atomic_json(RUN / "notes_sync_pending.json", {"error": str(error), "source_report_preserved": str(ROOT / "EVOMIND_TECHNICAL_REPORT.md")})
            if training_code or evaluation_code:
                raise SystemExit(training_code or evaluation_code)
            enable_vision_after_acceptance()
        else:
            enable_vision_after_acceptance()
        require_vision_enabled()
        raise SystemExit(run_manifest(MANIFEST, resume=args.resume or (ROOT / manifest['run_dir'] / "state.json").exists()))


def enable_vision_after_acceptance():
    from evomind_posttrain import RUN, verify_record, build_plan
    from evomind_run import sha256
    receipt = json.loads((RUN / "text_acceptance.json").read_text(encoding="utf-8"))
    if (receipt.get("status") != "accepted" or receipt.get("machine_evaluations_complete") is not True
            or receipt.get("human_review", {}).get("status") != "cancelled_by_user"):
        raise RuntimeError("Text acceptance is pending; vision gate remains closed")
    verify_record(receipt["base"])
    state = Path(receipt["evaluation_state"])
    if sha256(state) != receipt["evaluation_state_sha256"]:
        raise ValueError("Text evaluation state changed after acceptance")
    training = json.loads((RUN / "state.json").read_text(encoding="utf-8"))
    for branch in build_plan()["branches"]:
        row = training["branches"][branch["name"]]
        if row.get("status") != "completed":
            raise RuntimeError("A required text branch is incomplete")
        verify_record(row["output"])
    scope = json.loads(TEXT_SCOPE.read_text(encoding="utf-8"))
    scope.update(vision_enabled=True, text_acceptance=str(RUN / "text_acceptance.json"),
                 text_acceptance_sha256=sha256(RUN / "text_acceptance.json"))
    atomic_json(TEXT_SCOPE, scope)


if __name__ == "__main__":
    main()
