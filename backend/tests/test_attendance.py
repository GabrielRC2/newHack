from datetime import datetime, timezone
import unittest

from app.services.attendance import evaluate_exit_time


UTC = timezone.utc


class EvaluateExitTimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lesson_starts_at = datetime(2026, 9, 20, 14, 0, tzinfo=UTC)
        self.lesson_ends_at = datetime(2026, 9, 20, 16, 0, tzinfo=UTC)

    def evaluate_at(self, hour: int, minute: int) -> bool:
        result = evaluate_exit_time(
            lesson_starts_at=self.lesson_starts_at,
            lesson_ends_at=self.lesson_ends_at,
            occurred_at=datetime(2026, 9, 20, hour, minute, tzinfo=UTC),
        )
        return result.qualifies_for_attendance

    def test_before_halfway_only_identifies_the_student(self) -> None:
        self.assertFalse(self.evaluate_at(14, 20))
        self.assertFalse(self.evaluate_at(14, 50))

    def test_exactly_at_halfway_does_not_register_attendance(self) -> None:
        self.assertFalse(self.evaluate_at(15, 0))

    def test_after_halfway_registers_attendance(self) -> None:
        self.assertTrue(self.evaluate_at(15, 1))
        self.assertTrue(self.evaluate_at(15, 30))

    def test_timestamps_require_a_timezone(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_exit_time(
                lesson_starts_at=datetime(2026, 9, 20, 14, 0),
                lesson_ends_at=self.lesson_ends_at,
                occurred_at=datetime(2026, 9, 20, 15, 30, tzinfo=UTC),
            )


if __name__ == "__main__":
    unittest.main()
