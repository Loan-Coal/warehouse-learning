"""rl/spacetime.py: one robot's A* against a reservation table, on a corridor small enough to read."""
import unittest

from env.layout import parse
from env.warehouse import FORWARD, PICKUP, TURN_RIGHT
from rl.metrics import facing_field, facing_states, state_index
from rl.spacetime import Reservations, plan

GRID = parse("""
.......
.SSSSS.
.......
D.....D
""")
RIGHT, DOWN = 1, 2
TARGET = (3, 1)         # a shelf; its access cells are (3, 0) and (3, 2)


def search(start, heading, table=None, hard=(), soft=None, ready_at=lambda t: True, window=24,
           robot=0, target=TARGET):
    field = facing_field(GRID, target)
    goals = set(facing_states(GRID, target))
    return plan(GRID, robot, start, heading, ready_at,
                lambda c, h: state_index(GRID, (c, h)) in goals,
                lambda c, h: field[state_index(GRID, (c, h))],
                table or Reservations(), set(hard), soft or {}, window=window, finish=PICKUP)


class TestPlan(unittest.TestCase):
    def test_a_free_corridor_gives_the_shortest_plan_ending_in_the_pickup(self):
        path = search((0, 0), RIGHT)
        steps_to_face = facing_field(GRID, TARGET)[state_index(GRID, ((0, 0), RIGHT))]
        self.assertEqual(path[-1].action, PICKUP)
        self.assertEqual(len(path), steps_to_face + 1)

    def test_a_cell_booked_by_another_robot_is_avoided(self):
        free = search((0, 0), RIGHT)
        table = Reservations()
        table.book(1, free[0].cell, 1, 1)
        path = search((0, 0), RIGHT, table)
        self.assertNotEqual(path[0].cell, free[0].cell)

    def test_a_swap_is_refused(self):
        table = Reservations()
        table.book_move(1, (1, 0), (0, 0), 1)       # robot 1 arrives at (0, 0) from (1, 0) at tick 1
        path = search((0, 0), RIGHT, table)
        self.assertFalse(path[0].action == FORWARD and path[0].cell == (1, 0))

    def test_a_goal_out_of_the_window_gives_the_closest_partial_plan(self):
        path = search((0, 0), RIGHT, window=2)
        self.assertEqual(len(path), 2)
        self.assertNotEqual(path[-1].action, PICKUP)

    def test_a_slow_robot_waits_on_its_off_tick(self):
        # Already facing the target, but not ready now: PICKUP on tick 1 would be thrown away.
        path = search((3, 0), DOWN, ready_at=lambda t: t % 2 == 1)
        self.assertEqual([s.action for s in path], [TURN_RIGHT, PICKUP])
        self.assertEqual(path[0].cell, (3, 0))
        self.assertEqual(path[0].heading, DOWN)

    def test_an_unreachable_target_gives_none(self):
        self.assertIsNone(search((0, 0), RIGHT, target=(9, 9)))

    def test_release_drops_every_entry_of_a_robot(self):
        table = Reservations()
        table.book(1, (2, 0), 0, 3)
        table.book_move(1, (1, 0), (2, 0), 1)
        table.book(2, (4, 0), 1, 1)
        table.release(1)
        self.assertEqual(table.cells, {((4, 0), 1): 2})
        self.assertEqual(table.edges, {})
        self.assertEqual(table.crossed(), {(4, 0)})


if __name__ == "__main__":
    unittest.main()
