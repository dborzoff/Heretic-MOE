import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

from heretic.plugin import Context
from heretic.utils import Prompt


class FakeModel:
    def get_responses_batched(self, prompts, skip_special_tokens=True):
        del skip_special_tokens
        return [f"synthetic-{index}" for index, _ in enumerate(prompts)]


class ResponseArchiveTests(unittest.TestCase):
    def test_context_writes_one_atomic_archive_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "responses.sqlite3"
            settings = SimpleNamespace(
                save_trial_responses=True,
                trial_responses_file=str(archive),
            )
            model = FakeModel()
            context = Context(settings, model, response_archive_id=7)
            prompts = [Prompt(system="system", user="one"), Prompt(system="", user="two")]

            first = context.get_responses(prompts)
            second = context.get_responses(prompts)

            self.assertEqual(first, second)
            self.assertTrue(archive.is_file())
            with closing(sqlite3.connect(archive)) as database:
                prompt_count = database.execute(
                    "SELECT COUNT(*) FROM prompts"
                ).fetchone()[0]
                answer_rows = database.execute(
                    """
                    SELECT prompt_index, trial_id, trial_number, answer_sha256
                    FROM trial_answers ORDER BY prompt_index
                    """
                ).fetchall()
                grouped = database.execute(
                    "SELECT answer_count FROM answers_by_question ORDER BY prompt_index"
                ).fetchall()
            self.assertEqual(prompt_count, 2)
            self.assertEqual(len(answer_rows), 2)
            self.assertEqual([row[0] for row in answer_rows], [0, 1])
            self.assertTrue(all(row[1] == "trial:7" for row in answer_rows))
            self.assertTrue(all(row[2] == 7 for row in answer_rows))
            self.assertTrue(all(row[3] for row in answer_rows))
            self.assertEqual(grouped, [(1,), (1,)])

    def test_trial_number_offset_and_stride_support_parallel_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "responses.sqlite3"
            prompt = [Prompt(system="system", user="question")]
            for offset, local_trial in ((0, 3), (1, 3)):
                settings = SimpleNamespace(
                    save_trial_responses=True,
                    trial_responses_file=str(archive),
                    trial_response_number_offset=offset,
                    trial_response_number_stride=2,
                )
                Context(
                    settings,
                    FakeModel(),
                    response_archive_id=local_trial,
                ).get_responses(prompt)

            with closing(sqlite3.connect(archive)) as database:
                rows = database.execute(
                    "SELECT trial_id, trial_number FROM trial_answers "
                    "ORDER BY trial_number"
                ).fetchall()
            self.assertEqual(rows, [("trial:6", 6), ("trial:7", 7)])

    def test_two_branches_can_write_the_shared_archive_concurrently(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "responses.sqlite3"
            prompt = [Prompt(system="system", user="question")]

            def write(offset: int, local_trial: int) -> None:
                settings = SimpleNamespace(
                    save_trial_responses=True,
                    trial_responses_file=str(archive),
                    trial_response_number_offset=offset,
                    trial_response_number_stride=2,
                )
                Context(
                    settings,
                    FakeModel(),
                    response_archive_id=local_trial,
                ).get_responses(prompt)

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(write, offset, local_trial)
                    for local_trial in range(10)
                    for offset in (0, 1)
                ]
                for future in futures:
                    future.result()

            with closing(sqlite3.connect(archive)) as database:
                numbers = [
                    row[0]
                    for row in database.execute(
                        "SELECT trial_number FROM trial_answers "
                        "ORDER BY trial_number"
                    )
                ]
            self.assertEqual(numbers, list(range(20)))


if __name__ == "__main__":
    unittest.main()
