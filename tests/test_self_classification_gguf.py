from __future__ import annotations

import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def test_llama_client_applies_chat_template_before_completion() -> None:
    gguf = importlib.import_module("heretic.self_classification_gguf")
    calls: list[tuple[str, dict[str, object]]] = []

    def post_json(path: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append((path, payload))
        if path == "/apply-template":
            return {"prompt": "rendered-chat"}
        return {"content": "A", "tokens_predicted": 1}

    client = gguf.LlamaCompletionClient(post_json=post_json)
    text, tokens = client.generate(
        system="classifier-system",
        user="classifier-input",
        max_new_tokens=4,
    )

    assert (text, tokens) == ("A", 1)
    assert calls == [
        (
            "/apply-template",
            {
                "messages": [
                    {"role": "system", "content": "classifier-system"},
                    {"role": "user", "content": "classifier-input"},
                ]
            },
        ),
        (
            "/completion",
            {
                "prompt": "rendered-chat",
                "n_predict": 4,
                "temperature": 0.0,
                "stream": False,
            },
        ),
    ]


def test_server_command_disables_reasoning_and_keeps_model_on_selected_gpu(
    tmp_path: Path,
) -> None:
    gguf = importlib.import_module("heretic.self_classification_gguf")
    model = tmp_path / "model.gguf"
    server = tmp_path / "llama-server.exe"

    command, environment = gguf.build_server_process(
        server=server,
        model=model,
        device="1",
        port=8091,
        parallel=8,
        context_per_slot=2048,
        gpu_layers=999,
    )

    assert command[:4] == [str(server), "-m", str(model), "--port"]
    assert command[command.index("--port") + 1] == "8091"
    assert command[command.index("-np") + 1] == "8"
    assert command[command.index("-c") + 1] == "16384"
    assert command[command.index("-ngl") + 1] == "999"
    assert command[command.index("--reasoning") + 1] == "off"
    assert command[command.index("--reasoning-budget") + 1] == "0"
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"


def test_result_record_never_contains_prompt_or_generated_text() -> None:
    gguf = importlib.import_module("heretic.self_classification_gguf")
    from heretic.self_classification import ClassificationInput, PromptVariant

    row = ClassificationInput(
        canonical_id="P0001",
        row_id="EN-P0001",
        language="en",
        category_ids=("C01",),
        direction_class="unsafe",
        prompt="private-input",
    )

    result = gguf.make_result_record(
        model_id="model.gguf",
        row=row,
        variant=PromptVariant.CODE_PERMUTED,
        generated_text="A",
        output_tokens=1,
    )

    assert result["row_id"] == "EN-P0001"
    assert result["valid"] is True
    assert result["classification"] in {
        "DIRECT",
        "PARTIAL",
        "SOFT",
        "HARD_REFUSE",
    }
    assert not ({"prompt", "response", "raw_output", "text"} & result.keys())
    assert "private-input" not in str(result)
    assert "A" not in result.values()


def test_gguf_runner_resumes_without_repeating_completed_rows(tmp_path: Path) -> None:
    gguf = importlib.import_module("heretic.self_classification_gguf")
    from heretic.self_classification import ClassificationInput, PromptVariant

    rows = [
        ClassificationInput(
            canonical_id=f"P{index:04d}",
            row_id=f"EN-P{index:04d}",
            language="en",
            category_ids=("C01",),
            direction_class="unsafe",
            prompt=f"private-{index}",
        )
        for index in (1, 2)
    ]
    output = tmp_path / "rows.jsonl"
    completed = gguf.make_result_record(
        model_id="model.gguf",
        row=rows[0],
        variant=PromptVariant.CODE_PERMUTED,
        generated_text="A",
        output_tokens=1,
    )
    output.write_text(json.dumps(completed) + "\n", encoding="utf-8")

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, int]] = []

        def generate(
            self, *, system: str, user: str, max_new_tokens: int
        ) -> tuple[str, int]:
            self.calls.append((system, user, max_new_tokens))
            return "B", 1

    client = FakeClient()
    summary = gguf.classify_rows_with_gguf(
        client=client,
        model_id="model.gguf",
        rows=rows,
        variant=PromptVariant.CODE_PERMUTED,
        output_path=output,
        parallel=2,
    )

    written = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary == {"completed": 2, "generated": 1, "total": 2}
    assert len(client.calls) == 1
    assert len(written) == 2
    assert {row["row_id"] for row in written} == {"EN-P0001", "EN-P0002"}
    assert all(
        not ({"prompt", "response", "raw_output", "text"} & row.keys())
        for row in written
    )


def test_post_json_uses_real_http_boundary() -> None:
    gguf = importlib.import_module("heretic.self_classification_gguf")
    observed: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            observed["path"] = self.path
            observed["payload"] = json.loads(self.rfile.read(length))
            encoded = json.dumps({"content": "A", "tokens_predicted": 1}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = gguf.post_json(
            f"http://127.0.0.1:{server.server_port}",
            "/completion",
            {"prompt": "rendered", "n_predict": 4},
            timeout=5.0,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert result == {"content": "A", "tokens_predicted": 1}
    assert observed == {
        "path": "/completion",
        "payload": {"prompt": "rendered", "n_predict": 4},
    }


def test_finalize_gguf_results_verifies_coverage_and_writes_manifest(
    tmp_path: Path,
) -> None:
    gguf = importlib.import_module("heretic.self_classification_gguf")
    from heretic.self_classification import ClassificationInput, PromptVariant

    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    dataset_manifest = tmp_path / "dataset-manifest.json"
    dataset_manifest.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "result"
    output_dir.mkdir()
    rows = [
        ClassificationInput(
            canonical_id=f"P{index:04d}",
            row_id=f"EN-P{index:04d}",
            language="en",
            category_ids=("C01",),
            direction_class="unsafe",
            prompt=f"private-{index}",
        )
        for index in (1, 2)
    ]
    result_path = output_dir / "rows.jsonl"
    records = [
        gguf.make_result_record(
            model_id=model.name,
            row=row,
            variant=PromptVariant.CODE_PERMUTED,
            generated_text="A",
            output_tokens=1,
        )
        for row in rows
    ]
    result_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    manifest = gguf.finalize_gguf_results(
        output_dir=output_dir,
        model=model,
        dataset_manifest=dataset_manifest,
        model_id=model.name,
        rows=rows,
        variant=PromptVariant.CODE_PERMUTED,
    )

    assert manifest["status"] == "PASS"
    assert manifest["rows"] == 2
    assert manifest["valid"] == sum(record["valid"] for record in records)
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "model_summary.json").is_file()
    assert (output_dir / "report.html").is_file()
