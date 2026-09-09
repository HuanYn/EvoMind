"""CPU-only supervisor tests; child scripts never import model/GPU libraries."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import time
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("evomind_run", Path(__file__).resolve().parents[1] / "scripts" / "evomind_run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class LogParserTests(unittest.TestCase):
    def test_real_official_format(self):
        row = runner.parse_official_log("Epoch:[2/2](100/321), loss: 3.2100, logits_loss: 3.2000, aux_loss: 0.0100, lr: 0.00000123, epoch_time: 1.0min")
        self.assertEqual(row["global_microstep"], 421)
        self.assertEqual(row["microstep"], 100)
        self.assertAlmostEqual(row["learning_rate"], 1.23e-6)
        self.assertEqual(row["metric_scope"], "training_current_microbatch")
        self.assertFalse(row["nonfinite"])
        self.assertNotIn("val_loss", row)

    def test_scientific_bracket_and_nonfinite(self):
        row = runner.parse_official_log("Epoch:[1/2][9/10], loss: nan, logits_loss: inf, aux_loss: 0, lr: 1e-5")
        self.assertTrue(row["nonfinite"])
        self.assertIsNone(row["loss"])
        self.assertIsNone(runner.parse_official_log("Epoch 1 started"))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
    def test_real_agg_curve(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            folder = Path(temporary)
            metric = runner.parse_official_log("Epoch:[1/1](1/1), loss: 4.0, logits_loss: 4.0, aux_loss: 0.0, lr: 5e-5")
            runner.append_json(folder / "metrics.jsonl", metric)
            runner.render_curve(folder / "metrics.jsonl", folder / "curve.png")
            self.assertEqual((folder / "curve.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        (self.repo / "trainer").mkdir()
        (self.repo / "out").mkdir()
        self.run_dir = self.repo / "artifacts" / "evomind" / "test"
        self.manifest_path = self.repo / "manifest.json"
        self.plotter = patch.object(runner, "render_curve")
        self.plotter.start()
        self.addCleanup(self.plotter.stop)

    def script(self, name, content):
        path = self.repo / "trainer" / name
        path.write_text(content, encoding="utf-8")
        return path

    def manifest(self, stages):
        self.manifest_path.write_text(json.dumps({"run_dir": str(self.run_dir), "stages": stages}), encoding="utf-8")

    def run_queue(self, resume=False):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return runner.run_manifest(self.manifest_path, resume, self.repo)

    def read_state(self):
        return json.loads((self.run_dir / "state.json").read_text(encoding="utf-8"))

    def test_success_and_no_overwrite(self):
        self.script("first.py", "from pathlib import Path\nPath('../out/pretrain_768.pth').write_bytes(b'first checkpoint')\nprint('Epoch:[1/1](1/1), loss: 4.0, logits_loss: 4.0, aux_loss: 0.0, lr: 0.00005', end='')\n")
        self.script("second.py", "from pathlib import Path\nassert Path('../out/pretrain_768.pth').read_bytes() == b'first checkpoint'\nPath('../out/full_sft_768.pth').write_bytes(b'second checkpoint')\n")
        self.manifest([{"name": "pretrain", "argv": ["first.py"], "expected_weight": "pretrain"},
                       {"name": "sft", "argv": ["second.py"], "expected_weight": "full_sft"}])
        self.assertEqual(self.run_queue(), 0)
        state = self.read_state()
        self.assertEqual(state["status"], "completed")
        self.assertIsNone(state["child_pid"])
        self.assertTrue((self.run_dir / "snapshots" / "sft" / "full_sft_768.pth").is_file())
        metric = json.loads((self.run_dir / "metrics" / "pretrain.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(metric["microstep"], 1)
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            self.run_queue()
        self.assertEqual(self.run_queue(resume=True), 0)
        self.assertEqual(len((self.run_dir / "commands.jsonl").read_text(encoding="utf-8").splitlines()), 2)

    def test_failure_waits_for_exit_and_resume_skips_completed(self):
        self.script("first.py", "from pathlib import Path\nPath('../first-ran').write_text('yes')\n")
        self.script("fail_once.py", "from pathlib import Path\nimport sys\np=Path('../allow-success')\nif not p.exists():\n print('EOF is not success')\n sys.stdout.close()\n sys.exit(7)\nPath('../second-ran').write_text('yes')\n")
        self.script("third.py", "from pathlib import Path\nPath('../third-ran').write_text('yes')\n")
        self.manifest([{"name": "first", "argv": ["first.py"]},
                       {"name": "second", "argv": ["fail_once.py"]},
                       {"name": "third", "argv": ["third.py"]}])
        self.assertEqual(self.run_queue(), 1)
        self.assertEqual(self.read_state()["stages"]["second"]["exit_code"], 7)
        self.assertFalse((self.repo / "third-ran").exists())
        (self.repo / "allow-success").touch()
        self.assertEqual(self.run_queue(True), 0)
        commands = [json.loads(line) for line in (self.run_dir / "commands.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([item["stage"] for item in commands], ["first", "second", "second", "third"])

    def test_missing_expected_weight_stops_queue(self):
        self.script("empty.py", "print('nothing saved')\n")
        self.manifest([{"name": "empty", "argv": ["empty.py"], "expected_weight": "pretrain"},
                       {"name": "never", "argv": ["empty.py"]}])
        self.assertEqual(self.run_queue(), 1)
        self.assertNotIn("never", self.read_state()["stages"])

    def test_manifest_change_and_live_pid_refused(self):
        self.script("fail.py", "raise SystemExit(1)\n")
        self.manifest([{"name": "fail", "argv": ["fail.py"]}])
        self.assertEqual(self.run_queue(), 1)
        original = self.manifest_path.read_text(encoding="utf-8")
        self.manifest([{"name": "fail", "argv": ["fail.py", "--batch_size", "2"]}])
        with self.assertRaisesRegex(RuntimeError, "manifest hash"):
            self.run_queue(True)
        self.manifest_path.write_text(original, encoding="utf-8")
        state = self.read_state()
        state.update(status="running", pid=os.getpid())
        runner.atomic_json(self.run_dir / "state.json", state)
        with self.assertRaisesRegex(RuntimeError, "still running"):
            self.run_queue(True)

    def test_absolute_script_custom_cwd_and_required_input(self):
        script = self.script("custom.py", "from pathlib import Path\nassert Path('input.txt').read_text() == 'ready'\nprint('custom stage')\n")
        (self.repo / "input.txt").write_text("ready", encoding="utf-8")
        self.manifest([{"name": "custom", "argv": [str(script)], "cwd": str(self.repo), "requires": ["input.txt"]}])
        self.assertEqual(self.run_queue(), 0)

    def test_official_resume_requires_owned_checkpoint(self):
        self.script("train_pretrain.py", "raise SystemExit(3)\n")
        self.manifest([{"name": "pretrain", "argv": ["train_pretrain.py"]}])
        self.assertEqual(self.run_queue(), 1)
        self.assertEqual(self.run_queue(True), 1)
        self.assertIn("no verified checkpoint", self.read_state()["error"])

    def test_official_resume_adds_flag_for_matching_checkpoint(self):
        self.script("train_pretrain.py", "from pathlib import Path\nimport sys\np=Path('../checkpoints/pretrain_768_resume.pth')\np.parent.mkdir(exist_ok=True)\nif '--from_resume' not in sys.argv:\n p.write_bytes(b'fake CPU checkpoint')\n raise SystemExit(3)\nassert sys.argv[sys.argv.index('--from_resume')+1] == '1'\n")
        self.manifest([{"name": "pretrain", "argv": ["train_pretrain.py"]}])
        self.assertEqual(self.run_queue(), 1)
        self.assertEqual(self.run_queue(True), 0)
        self.assertEqual(self.read_state()["stages"]["pretrain"]["argv"][-2:], ["--from_resume", "1"])

    def test_repeated_resume_preserves_verified_checkpoint(self):
        self.script("train_pretrain.py", "from pathlib import Path\nimport sys\np=Path('../checkpoints/pretrain_768_resume.pth')\np.parent.mkdir(exist_ok=True)\nif '--from_resume' not in sys.argv:\n p.write_bytes(b'fake CPU checkpoint')\nif not Path('../allow-success').exists():\n raise SystemExit(3)\n")
        self.manifest([{"name": "pretrain", "argv": ["train_pretrain.py"]}])
        self.assertEqual(self.run_queue(), 1)
        self.assertEqual(self.run_queue(True), 1)
        (self.repo / "allow-success").touch()
        self.assertEqual(self.run_queue(True), 0)

    def test_nonfinite_metric_terminates_child(self):
        self.script("nonfinite.py", "import time\nprint('Epoch:[1/1](1/5), loss: nan, logits_loss: nan, aux_loss: 0, lr: 5e-5', flush=True)\ntime.sleep(30)\n")
        self.manifest([{"name": "bad", "argv": ["nonfinite.py"]}])
        self.assertEqual(self.run_queue(), 1)
        state = self.read_state()
        self.assertIn("nonfinite", state["error"])
        self.assertFalse(runner.pid_alive(state["stages"]["bad"]["child_pid"]))

    def test_os_lock_refuses_duplicate_supervisor(self):
        self.run_dir.mkdir(parents=True)
        with runner.exclusive_run_lock(self.run_dir):
            with self.assertRaisesRegex(RuntimeError, "active supervisor"):
                with runner.exclusive_run_lock(self.run_dir):
                    self.fail("lock must not be acquired")

    def test_nested_continuation_resume_requires_inner_state(self):
        (self.repo / "configs").mkdir()
        (self.repo / "configs/vision_pipeline.json").write_text(
            json.dumps({"run_dir": "artifacts/inner"}), encoding="utf-8")
        self.script("evomind_continue.py", "from pathlib import Path\nimport sys\np=Path('../artifacts/inner/state.json')\nif not p.exists():\n p.parent.mkdir(parents=True,exist_ok=True)\n p.write_text('{}')\n raise SystemExit(4)\nassert '--resume' in sys.argv\n")
        self.manifest([{"name": "continuation", "argv": ["evomind_continue.py"]}])
        self.assertEqual(self.run_queue(), 1)
        self.assertEqual(self.run_queue(True), 0)
        self.assertIn("--resume", self.read_state()["stages"]["continuation"]["argv"])

    def test_nested_resume_not_added_without_inner_state(self):
        (self.repo / "configs").mkdir()
        (self.repo / "configs/vision_pipeline.json").write_text(
            json.dumps({"run_dir": "artifacts/inner"}), encoding="utf-8")
        argv = ["scripts/evomind_continue.py"]
        self.assertEqual(runner.continuation_resume_argv(argv, self.repo, True), argv)
        inner = self.repo / "artifacts/inner"
        inner.mkdir(parents=True)
        (inner / "state.json").write_text("{}", encoding="utf-8")
        self.assertEqual(runner.continuation_resume_argv(argv, self.repo, False), argv)


@unittest.skipUnless(os.name == "nt", "Windows Job Object regression tests")
class WindowsProcessTreeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.pids_path = self.folder / "tree_pids.json"
        self.leaf_pid = self.folder / "leaf_pid.txt"
        self.leaf = self.folder / "leaf.py"
        self.branch = self.folder / "branch.py"
        self.leaf.write_text(
            "import os,sys,time\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(str(os.getpid()))\ntime.sleep(20)\n",
            encoding="utf-8")
        self.branch.write_text(
            "import json,os,subprocess,sys,time\nfrom pathlib import Path\n"
            f"p=subprocess.Popen([sys.executable,'-u',{str(self.leaf)!r},{str(self.leaf_pid)!r}])\n"
            f"leaf=Path({str(self.leaf_pid)!r})\n"
            "deadline=time.monotonic()+8\nwhile not leaf.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
            "assert leaf.exists(),'leaf did not start'\n"
            f"Path({str(self.pids_path)!r}).write_text(json.dumps([os.getpid(),p.pid,int(leaf.read_text())]))\n"
            "print('TREE_READY',flush=True)\ntime.sleep(20)\n", encoding="utf-8")

    def assert_tree_gone(self, additional=()):
        pids = json.loads(self.pids_path.read_text(encoding="utf-8")) + list(additional)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(runner.pid_alive(pid) for pid in pids):
            time.sleep(0.02)
        self.assertEqual([pid for pid in pids if runner.pid_alive(pid)], [],
                         "A test-owned redirector/Python descendant survived cleanup")

    def test_keyboard_interrupt_terminates_redirector_and_grandchildren(self):
        with self.assertRaises(KeyboardInterrupt):
            with runner.managed_child([sys.executable, "-u", str(self.branch)],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, encoding="utf-8") as child:
                launcher_pid = child.pid
                self.assertEqual(child.stdout.readline().strip(), "TREE_READY")
                raise KeyboardInterrupt()
        self.assert_tree_gone([launcher_pid])

    def test_owner_abrupt_exit_closes_inner_job_before_outer_job(self):
        owner = self.folder / "owner.py"
        owner_pid_path = self.folder / "owned_launcher.txt"
        scripts = Path(runner.__file__).resolve().parent
        owner.write_text(
            "import os,subprocess,sys,time\nfrom pathlib import Path\n"
            f"sys.path.insert(0,{str(scripts)!r})\nfrom evomind_run import managed_child\n"
            f"with managed_child([sys.executable,'-u',{str(self.branch)!r}],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) as child:\n"
            f" Path({str(owner_pid_path)!r}).write_text(str(child.pid))\n"
            " deadline=time.monotonic()+8\n"
            f" while not Path({str(self.pids_path)!r}).exists() and time.monotonic()<deadline: time.sleep(.01)\n"
            f" assert Path({str(self.pids_path)!r}).exists(),'tree did not start'\n"
            " os._exit(23)\n", encoding="utf-8")
        with runner.managed_child([sys.executable, "-u", str(owner)],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8") as process:
            output, _ = process.communicate(timeout=12)
            self.assertEqual(process.returncode, 23, output)
            self.assert_tree_gone([int(owner_pid_path.read_text(encoding="utf-8"))])


if __name__ == "__main__":
    unittest.main()
