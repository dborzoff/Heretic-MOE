# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest

import torch
import torch.nn.functional as F

from heretic.config import RowNormalization
from heretic.model import low_rank_frobenius_squared, project_fused_expert_chunk


class FusedProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.weights = torch.randn(5, 7, 11, dtype=torch.float32)
        self.direction = F.normalize(torch.randn(7), dim=0)

    def test_none_matches_direct_projection(self) -> None:
        actual = project_fused_expert_chunk(
            self.weights,
            self.direction,
            1.25,
            RowNormalization.NONE,
        )
        projection = torch.einsum("h,ehi->ei", self.direction, self.weights)
        expected = self.weights - 1.25 * self.direction.view(
            1, -1, 1
        ) * projection.unsqueeze(1)
        torch.testing.assert_close(actual, expected)

    def test_full_preserves_every_row_norm(self) -> None:
        actual = project_fused_expert_chunk(
            self.weights,
            self.direction,
            1.6,
            RowNormalization.FULL,
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(actual, dim=2),
            torch.linalg.vector_norm(self.weights, dim=2),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_chunking_does_not_change_result(self) -> None:
        whole = project_fused_expert_chunk(
            self.weights,
            self.direction,
            0.7,
            RowNormalization.PRE,
        )
        chunked = torch.cat(
            [
                project_fused_expert_chunk(
                    self.weights[start : start + 2],
                    self.direction,
                    0.7,
                    RowNormalization.PRE,
                )
                for start in range(0, len(self.weights), 2)
            ]
        )
        torch.testing.assert_close(chunked, whole)

    def test_low_rank_norm_matches_materialized_delta(self) -> None:
        left = torch.randn(7, 3)
        right = torch.randn(3, 11)
        expected = float(torch.sum((left @ right) ** 2))
        self.assertAlmostEqual(
            low_rank_frobenius_squared(left, right),
            expected,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
