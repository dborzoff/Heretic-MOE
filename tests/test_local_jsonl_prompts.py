import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from heretic.config import DatasetSpecification
from heretic.utils import load_prompts


class LocalJsonlPromptTests(unittest.TestCase):
    def test_uses_configured_column_instead_of_whole_json_line(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "prompts.jsonl"
            rows = [
                {"prompt": "synthetic one", "metadata": 1},
                {"prompt": "synthetic two", "metadata": 2},
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            settings = SimpleNamespace(system_prompt="system")
            specification = DatasetSpecification(
                dataset=str(source),
                column="prompt",
            )

            prompts = load_prompts(settings, specification)

            self.assertEqual([prompt.user for prompt in prompts], ["synthetic one", "synthetic two"])
            self.assertTrue(all(prompt.system == "system" for prompt in prompts))

    def test_jsonl_requires_a_column(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "prompts.jsonl"
            source.write_text('{"prompt":"synthetic"}\n', encoding="utf-8")
            settings = SimpleNamespace(system_prompt="")

            with self.assertRaises(ValueError):
                load_prompts(settings, DatasetSpecification(dataset=str(source)))


if __name__ == "__main__":
    unittest.main()
