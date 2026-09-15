import unittest

from src.Core.CrossSection.cross_section import CrossSection
from src.Core.CrossSection.schedule import SectionSchedule, SectionSpan


class SectionSpanTests(unittest.TestCase):
    def test_a_single_point_span_is_allowed(self) -> None:
        span = SectionSpan(4, 4, CrossSection())
        self.assertTrue(span.contains(4))
        self.assertFalse(span.contains(3))
        self.assertFalse(span.contains(5))

    def test_both_ends_are_inclusive(self) -> None:
        span = SectionSpan(2, 5, CrossSection())
        self.assertTrue(span.contains(2))
        self.assertTrue(span.contains(5))

    def test_negative_start_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SectionSpan(-1, 3, CrossSection())

    def test_stop_before_start_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SectionSpan(3, 2, CrossSection())


class SectionScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.road = CrossSection()
        self.bridge = CrossSection()

    def test_spans_are_sorted_by_start(self) -> None:
        later = SectionSpan(6, 9, self.bridge)
        earlier = SectionSpan(0, 3, self.road)
        schedule = SectionSchedule([later, earlier], path_length=10)
        self.assertEqual(schedule.spans, (earlier, later))

    def test_at_finds_the_section_or_none_in_a_gap(self) -> None:
        schedule = SectionSchedule(
            [SectionSpan(0, 3, self.road), SectionSpan(6, 9, self.bridge)],
            path_length=10,
        )
        self.assertIs(schedule.at(0), self.road)
        self.assertIs(schedule.at(3), self.road)
        self.assertIsNone(schedule.at(4))
        self.assertIsNone(schedule.at(5))
        self.assertIs(schedule.at(6), self.bridge)
        self.assertIs(schedule.at(9), self.bridge)

    def test_at_is_none_before_the_first_span(self) -> None:
        schedule = SectionSchedule([SectionSpan(2, 4, self.road)], path_length=5)
        self.assertIsNone(schedule.at(0))

    def test_an_empty_schedule_delivers_nothing(self) -> None:
        schedule = SectionSchedule([], path_length=5)
        self.assertEqual(schedule.spans, ())
        self.assertIsNone(schedule.at(0))

    def test_adjacent_spans_are_allowed(self) -> None:
        schedule = SectionSchedule(
            [SectionSpan(0, 3, self.road), SectionSpan(4, 9, self.bridge)],
            path_length=10,
        )
        self.assertIs(schedule.at(3), self.road)
        self.assertIs(schedule.at(4), self.bridge)

    def test_overlapping_spans_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SectionSchedule(
                [SectionSpan(0, 4, self.road), SectionSpan(4, 6, self.bridge)],
                path_length=10,
            )

    def test_a_span_past_the_path_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SectionSchedule([SectionSpan(0, 10, self.road)], path_length=10)
        SectionSchedule([SectionSpan(0, 9, self.road)], path_length=10)


if __name__ == "__main__":
    unittest.main()
