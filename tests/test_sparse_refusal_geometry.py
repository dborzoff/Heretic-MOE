import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from heretic.config import DatasetSpecification, Settings as HereticSettings
from heretic.scorers.sparse_refusal_geometry import (
    Settings,
    SparseRefusalGeometry,
)
from heretic.utils import Prompt


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": text.split()}

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(ids)


class _Context:
    def __init__(self, prompts, responses):
        self._prompts = prompts
        self._responses = responses
        self._model = SimpleNamespace(tokenizer=_Tokenizer())

    def load_prompts(self, specification):
        del specification
        return self._prompts

    def get_responses(self, prompts):
        assert prompts == self._prompts
        return self._responses


class SparseRefusalGeometryTests(unittest.TestCase):
    def test_score_is_finite_and_empty_response_is_penalized(self):
        prompts = [Prompt(system="", user=f"prompt {index}") for index in range(6)]
        labels = ["delivered", "delivered", "soft", "soft", "refuse", "refuse"]
        answers = [
            "direct useful response",
            "specific completed result",
            "safer alternative instead",
            "redirected general discussion",
            "cannot provide that",
            "unable to assist",
        ]
        responses = ["direct useful response", "", "alternative", "", "cannot", ""]
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "prototypes.jsonl"
            with bank.open("w", encoding="utf-8") as handle:
                for index, (label, answer) in enumerate(zip(labels, answers)):
                    handle.write(
                        json.dumps(
                            {
                                "id": index,
                                "prompt": prompts[index].user,
                                "answer": answer,
                                "label": label,
                            }
                        )
                        + "\n"
                    )
            scorer = SparseRefusalGeometry(
                heretic_settings=HereticSettings(
                    model="unused", max_response_length=8
                ),
                settings=Settings(
                    prototypes=str(bank),
                    prompts=DatasetSpecification(dataset="unused"),
                    top_k=1,
                    min_df=1,
                    char_max_features=1_000,
                    word_max_features=1_000,
                ),
            )
            context = _Context(prompts, responses)
            scorer.init(context)
            score = scorer.get_score(context)

        self.assertEqual(score.diagnostics["rows"], 6)
        self.assertEqual(score.diagnostics["empty_indices"], [1, 3, 5])
        self.assertEqual(len(score.diagnostics["margins"]), 6)
        self.assertGreater(score.value, 0.0)


if __name__ == "__main__":
    unittest.main()
