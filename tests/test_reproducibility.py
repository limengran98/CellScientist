"""Regression checks for API portability and a frozen, CPU-backed run."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import h5py
import numpy as np

from cellscientist.cli import main as cli
from cellscientist.data import load_task_data
from cellscientist.llm_client import OpenAICompatiblePolicy
from cellscientist.protocol import load_config, task_specs, verify_lock, write_lock
from cellscientist.runner import run_one
from cellscientist.schemas import ContractError
from scripts.configure_run import main as configure
from scripts.download_data import download

ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def mock_endpoint(*, usage=None, malformed=False):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "body": body, "authorization": self.headers.get("Authorization")})
            prompt = body["messages"][-1]["content"]
            outputs = next((line.split("=", 1)[1] for line in prompt.splitlines() if line.startswith("valid_outputs=")), None)
            choice = json.loads(outputs)[0] if outputs else {"candidate_id": "health", "address": "conditioning"}
            content = "not a decision" if malformed else json.dumps(choice)
            response = json.dumps({"choices": [{"message": {"content": content}}], "usage": usage}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def llm_config(endpoint):
    return {"enabled": True, "model": "test-model", "base_url": endpoint,
            "api_key_env": "CELLSCIENTIST_TEST_KEY", "require_credential": False,
            "max_attempts": 1, "selection_status": "frozen", "timeout_seconds": 5}


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"CELLSCIENTIST_TEST_KEY": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_local_keyless_request_and_null_usage(self):
        with mock_endpoint() as (endpoint, requests):
            policy = OpenAICompatiblePolicy(llm_config(endpoint + "/"))
            decision = policy.choose("health", ["health"], ["conditioning"])
        self.assertEqual(requests[0]["path"], "/v1/chat/completions")
        self.assertIsNone(requests[0]["authorization"])
        self.assertEqual(requests[0]["body"]["model"], "test-model")
        self.assertEqual(decision.usage["total_tokens"], 0)
        self.assertFalse(decision.http_attempts[0]["usage_reported"])

    def test_hosted_auth_and_alternate_request_parameters(self):
        with mock_endpoint(usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}) as (endpoint, requests):
            config = {**llm_config(endpoint), "require_credential": True,
                      "temperature": None, "max_tokens_parameter": "max_completion_tokens", "max_tokens": 123}
            with patch.dict(os.environ, {"CELLSCIENTIST_TEST_KEY": "test-only-key"}):
                decision = OpenAICompatiblePolicy(config).choose("health", ["health"], ["conditioning"])
        self.assertEqual(requests[0]["authorization"], "Bearer test-only-key")
        self.assertEqual(requests[0]["body"]["max_completion_tokens"], 123)
        self.assertNotIn("temperature", requests[0]["body"])
        self.assertNotIn("max_tokens", requests[0]["body"])
        self.assertEqual(decision.usage["total_tokens"], 7)

    def test_required_key_is_checked_before_http(self):
        policy = OpenAICompatiblePolicy({**llm_config("http://unused.invalid"), "require_credential": True})
        with patch("urllib.request.urlopen") as transport:
            with self.assertRaisesRegex(ContractError, "Missing credential"):
                policy.choose("health", ["health"], ["conditioning"])
            transport.assert_not_called()

    def test_malformed_decision_retains_trace(self):
        with mock_endpoint(malformed=True) as (endpoint, _):
            policy = OpenAICompatiblePolicy(llm_config(endpoint))
            with self.assertRaisesRegex(ContractError, "JSON object"):
                policy.choose("health", ["health"], ["conditioning"])
        self.assertEqual(len(policy.last_call_trace), 1)
        self.assertTrue(policy.last_call_trace[0]["http_success"])
        self.assertIsNotNone(policy.last_validation_error)

    def test_invalid_usage_is_a_recorded_contract_failure(self):
        with mock_endpoint(usage={"total_tokens": "invalid"}) as (endpoint, _):
            policy = OpenAICompatiblePolicy(llm_config(endpoint))
            with self.assertRaises(ContractError):
                policy.choose("health", ["health"], ["conditioning"])
        self.assertEqual(policy.last_call_trace[0]["error_type"], "ValueError")


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.h5 = self.folder / "fixture.h5"
        rng = np.random.default_rng(101)
        pre = rng.normal(size=(60, 8)).astype("float32")
        with h5py.File(self.h5, "w") as handle:
            group = handle.create_group("combined")
            group["morphology_pre"] = pre
            group["morphology_post"] = pre + rng.normal(0, 0.2, pre.shape).astype("float32")
            group["dose"] = np.ones((60, 1), dtype="float32")
            group["smiles"] = np.asarray([f"C{i}".encode() for i in range(60)])
            group["plate_id"] = np.asarray([f"P{i}".encode() for i in range(60)])
            group["split_id"] = np.repeat(np.arange(1, 6), 12)
        self.config = json.loads((ROOT / "configs/bbbc036_047_formal.json").read_text())
        self.config["protocol_id"] = "synthetic_regression"
        self.config["data"]["tasks"] = [{"dataset": "Fixture", "split": "smiles", "path": str(self.h5), "group_key": "smiles"}]
        self.config["model"]["backend"] = "sklearn"
        self.config["search"].update(budget=2, budget_checkpoints=[1, 2], wallclock_checkpoints_seconds=[])
        self.config["outputs"] = {"root": str(self.folder / "results"), "cache": str(self.folder / "cache")}
        self.config["llm"] = llm_config("http://unused.invalid/v1")
        self.path = self.folder / "config.json"
        self.save_config()

    def save_config(self):
        self.path.write_text(json.dumps(self.config), encoding="utf-8")

    def load_data(self):
        config = load_config(self.path)
        return load_task_data(next(task_specs(config)), config["data"], config["search"])

    def test_partition_shapes_and_group_separation(self):
        data = self.load_data()
        self.assertEqual(data.dose.shape, (60,))
        self.assertEqual(data.partition_summary["partition_sizes"], {"train": 36, "feedback": 6, "selection": 6, "test": 12})
        groups = [set(data.smiles[list(getattr(data.partitions, key))]) for key in ("train", "feedback", "selection", "test")]
        for i, group in enumerate(groups):
            for other in groups[i + 1:]:
                self.assertFalse(group & other)

    def test_group_leakage_is_rejected(self):
        with h5py.File(self.h5, "a") as handle:
            handle["combined/smiles"][59] = b"C0"
        with self.assertRaisesRegex(ContractError, "Group leakage"):
            self.load_data()

    def test_empty_test_partition_is_rejected(self):
        with h5py.File(self.h5, "a") as handle:
            handle["combined/split_id"][48:] = 1
        with self.assertRaisesRegex(ContractError, "Training and test"):
            self.load_data()

    def test_split_group_mismatch_is_rejected(self):
        self.config["data"]["tasks"][0]["group_key"] = "plate_id"
        self.save_config()
        with self.assertRaisesRegex(ContractError, "group_key=smiles"):
            load_config(self.path)

    def test_lock_detects_config_and_dataset_changes(self):
        lock_path = self.folder / "protocol.lock.json"
        write_lock(self.path, lock_path, ROOT)
        verify_lock(self.path, lock_path, ROOT)
        self.config["llm"]["model"] = "another-model"
        self.save_config()
        with self.assertRaisesRegex(ContractError, "lock mismatch"):
            verify_lock(self.path, lock_path, ROOT)
        self.config["llm"]["model"] = "test-model"
        self.save_config()
        with h5py.File(self.h5, "a") as handle:
            handle["combined/morphology_post"][0, 0] += 1
        with self.assertRaisesRegex(ContractError, "lock mismatch"):
            verify_lock(self.path, lock_path, ROOT)

    def test_lock_detects_source_changes(self):
        source = self.folder / "cellscientist" / "module.py"
        source.parent.mkdir()
        source.write_text("version = 1\n")
        lock_path = self.folder / "protocol.lock.json"
        write_lock(self.path, lock_path, self.folder)
        source.write_text("version = 2\n")
        with self.assertRaisesRegex(ContractError, "lock mismatch"):
            verify_lock(self.path, lock_path, self.folder)

    def test_preflight_returns_failure_for_missing_cuda(self):
        self.config["model"]["backend"] = "torch_cuda"
        self.config["llm"]["enabled"] = False
        self.save_config()
        with patch.dict("sys.modules", {"torch": None}), contextlib.redirect_stdout(io.StringIO()) as output:
            status = cli(["preflight", "--config", str(self.path)])
        self.assertEqual(status, 1)
        self.assertFalse(json.loads(output.getvalue())["ready"])

    def test_frozen_cpu_trajectory_with_keyless_llm(self):
        with mock_endpoint() as (endpoint, requests), patch.dict(os.environ, {"CELLSCIENTIST_TEST_KEY": ""}):
            self.config["llm"] = llm_config(endpoint)
            self.save_config()
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli(["preflight", "--config", str(self.path), "--check-llm"]), 0)
            self.assertTrue(json.loads(output.getvalue())["ready"])
            lock_path = self.folder / "protocol.lock.json"
            lock = write_lock(self.path, lock_path, ROOT)
            kwargs = dict(root=ROOT, config_path=self.path, task_id="Fixture_smiles", controller="cellscientist", initialization="standard_h0", seed=11, lock_path=lock_path)
            result = run_one(**kwargs)
            reused = run_one(**kwargs)
        self.assertEqual(result["protocol_sha256"], lock["lock_sha256"])
        self.assertEqual(result["evaluated_candidates"], 2)
        self.assertEqual(result["llm_valid_decisions"], 1)
        self.assertEqual(result["llm_fallbacks"], 0)
        self.assertTrue(np.isfinite(result["test_metrics"]["pcc"]))
        self.assertEqual(result["selected_candidate_id"], result["budget_checkpoints"]["2"]["candidate_id"])
        self.assertEqual(reused["selected_candidate_id"], result["selected_candidate_id"])
        self.assertEqual(len(requests), 2)  # health + one proposal; repeat reuses the completed run
        run_dir = self.folder / "results/Fixture_smiles/standard_h0/cellscientist/seed-11"
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        self.assertIn("numpy", manifest["runtime"]["packages"])
        self.assertEqual(len((run_dir / "llm_trace.jsonl").read_text().splitlines()), 1)

    def test_routing_audit_uses_llm_without_loading_hdf5(self):
        with mock_endpoint() as (endpoint, requests):
            self.config["llm"] = llm_config(endpoint)
            self.config["data"]["tasks"][0]["path"] = "absent.h5"
            self.save_config()
            output_path = self.folder / "routing.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli(["routing-audit", "--config", str(self.path), "--output", str(output_path)]), 0)
        result = json.loads(output_path.read_text())
        self.assertEqual(result["routing_mode"], "llm_constrained")
        self.assertEqual(len(requests), 15)
        self.assertEqual(result["fallback_rate"], 0)

    def test_initialization_failure_saves_run_provenance(self):
        self.config["llm"].update(require_credential=True)
        self.save_config()
        lock_path = self.folder / "protocol.lock.json"
        write_lock(self.path, lock_path, ROOT)
        with patch.dict(os.environ, {"CELLSCIENTIST_TEST_KEY": ""}):
            with self.assertRaisesRegex(ContractError, "Missing credential"):
                run_one(root=ROOT, config_path=self.path, task_id="Fixture_smiles", controller="cellscientist", initialization="standard_h0", seed=11, lock_path=lock_path)
        failure_path = self.folder / "results/Fixture_smiles/standard_h0/cellscientist/seed-11/run_failure.json"
        failure = json.loads(failure_path.read_text())
        self.assertEqual(failure["error_type"], "ContractError")
        self.assertEqual(failure["task_id"], "Fixture_smiles")


class SetupTests(unittest.TestCase):
    def test_named_config_preserves_protocol_and_separates_outputs(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / "qwen.json"
            configure(["--base", str(ROOT / "configs/bbbc036_047_formal.json"), "--name", "qwen", "--model", "Qwen/Qwen2.5-0.5B-Instruct", "--local-api", "--backend", "sklearn", "--output", str(path)])
            config = json.loads(path.read_text())
            base = json.loads((ROOT / "configs/bbbc036_047_formal.json").read_text())
            self.assertEqual(config["search"], base["search"])
            self.assertEqual(config["data"], base["data"])
            self.assertEqual(config["outputs"]["root"], "outputs/qwen")
            self.assertFalse(config["llm"]["require_credential"])
            self.assertEqual(config["model"]["backend"], "sklearn")

    def test_download_verifies_arranges_and_reuses_files(self):
        content = b"synthetic download fixture"
        record = {"dataset": "Fixture", "runtime_path": "Fixture/data.h5", "hub_path": "source/data.h5", "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        manifest = {"repo_id": "test/fixture", "revision": "pinned-revision", "files": [record]}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            calls = []
            def fetch(**kwargs):
                calls.append(kwargs)
                path = root / "download.h5"
                path.write_bytes(content)
                return path
            download(root, manifest, ["Fixture"], fetch)
            download(root, manifest, ["Fixture"], fetch)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["revision"], "pinned-revision")
            self.assertEqual((root / "Fixture/data.h5").read_bytes(), content)
            receipt = json.loads((root / "download_manifest.json").read_text())
            self.assertEqual(receipt["files"][0]["sha256"], record["sha256"])
            (root / "Fixture/data.h5").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "different checksum"):
                download(root, manifest, ["Fixture"], fetch)

    def test_corrupt_download_is_rejected_before_installing(self):
        record = {"dataset": "Fixture", "runtime_path": "Fixture/data.h5", "hub_path": "source/data.h5", "size_bytes": 5, "sha256": hashlib.sha256(b"right").hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "download.h5"
            source.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "verification"):
                download(root, {"repo_id": "test/fixture", "revision": "pinned", "files": [record]}, ["Fixture"], lambda **_: source)
            self.assertFalse((root / "Fixture/data.h5").exists())


if __name__ == "__main__":
    unittest.main()
