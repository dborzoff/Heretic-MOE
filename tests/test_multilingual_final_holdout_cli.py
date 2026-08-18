from heretic.multilingual_final_holdout_cli import _autotune_final_holdout_batch


def test_final_holdout_worker_autotunes_zero_batch_before_generation() -> None:
    calls = []

    class FakeModel:
        def autotune_generation_batch_size(self, prompts, *, expected_rows):
            calls.append((list(prompts), expected_rows))
            return {"status": "PASS", "batch_size": 8}

    prompts = [object(), object(), object()]

    result = _autotune_final_holdout_batch(FakeModel(), prompts, 0)

    assert result == {"status": "PASS", "batch_size": 8}
    assert calls == [(prompts, 3)]


def test_final_holdout_worker_keeps_explicit_batch() -> None:
    class FakeModel:
        def autotune_generation_batch_size(self, prompts, *, expected_rows):
            raise AssertionError("explicit batch must not be autotuned")

    assert _autotune_final_holdout_batch(FakeModel(), [object()], 4) is None
