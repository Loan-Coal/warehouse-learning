"""rl/metrics.py measures how a fleet moves, whatever policy drives it.

The numbers are only worth logging if they count exactly what they claim, so every test here
scripts the actions by hand on a map small enough to work the right answer out on paper.
"""
import unittest

from env.warehouse import (BACKWARD, DOWN, DROPOFF, FORWARD, LEFT, PICKUP, RIGHT, TURN_LEFT,
                           TURN_RIGHT, UP, Task, Warehouse)
from rl.metrics import COUNTS, EpisodeMetrics, job_optimum, rates

# A shelf at the west end of a corridor, a dock at the east end, open rows around it.
CORRIDOR = """
......
S....D
......
"""


def scripted(*placements, layout=CORRIDOR, **kwargs):
    """An env with no new jobs and robots placed exactly; placements are (pos, heading)."""
    env = Warehouse(layout, n_robots=len(placements), alpha=0.0, task_prob=0.0, **kwargs)
    env.reset(0)
    for robot, (pos, heading) in zip(env.robots, placements):
        robot.pos, robot.heading = pos, heading
    return env


def give_retrieve(env, robot=0):
    """A retrieve job from the corridor's shelf (0, 1) to its dock (5, 1)."""
    env.crate_at[(0, 1)] = 99
    env.crate_info[99] = (1, 1)
    env.queue.append(env._new_task("retrieve"))
    env._assign_idle()
    return env.robots[robot].task


def play(env, script):
    """Run per-step action lists through env and a fresh tracker; returns the tracker's totals."""
    metrics = EpisodeMetrics()
    metrics.reset(env)
    for actions in script:
        metrics.observe(env, actions)
        env.step(actions)
        metrics.record(env)
    return metrics.totals()


class TestJobOptimum(unittest.TestCase):
    def test_turns_count_and_backing_up_is_as_good_as_turning_round(self):
        env = scripted(((1, 1), LEFT))
        task = give_retrieve(env)
        self.assertEqual((task.pickup, task.dropoff), ((0, 1), (5, 1)))
        # Already facing the shelf: pick up (1), back up 3 cells (3), turn round (2), drop off (1).
        self.assertEqual(job_optimum(env.grid, ((1, 1), LEFT), task.pickup, task.dropoff), 7)

    def test_facing_away_costs_the_turns(self):
        env = scripted(((1, 1), RIGHT))
        task = give_retrieve(env)
        # Turn round to face the shelf (2), pick up (1), back up 3 (3), turn round (2), drop off (1).
        # Reaching the shelf from above instead costs 5 before the pickup, so 9 is the optimum.
        self.assertEqual(job_optimum(env.grid, ((1, 1), RIGHT), task.pickup, task.dropoff), 9)


class TestJobs(unittest.TestCase):
    def test_an_optimal_scripted_job_scores_exactly_one(self):
        env = scripted(((1, 1), LEFT))
        give_retrieve(env)
        script = [[PICKUP], [BACKWARD], [BACKWARD], [BACKWARD], [TURN_RIGHT], [TURN_RIGHT], [DROPOFF]]
        totals = play(env, script)
        self.assertEqual(env.totals["deliveries"], 1)
        self.assertEqual((totals["job_steps"], totals["job_optimal"]), (7, 7))

    def test_a_detour_costs_its_extra_steps(self):
        env = scripted(((1, 1), LEFT))
        give_retrieve(env)
        script = [[PICKUP], [BACKWARD], [FORWARD], [BACKWARD], [BACKWARD], [BACKWARD],
                  [TURN_RIGHT], [TURN_RIGHT], [DROPOFF]]
        totals = play(env, script)
        self.assertEqual((totals["job_steps"], totals["job_optimal"]), (9, 7))

    def test_a_slow_robot_is_measured_in_its_own_actions(self):
        # Robot 1 is the slow one under mixed_robots; robot 0 just turns in place out of the way.
        env = scripted(((1, 0), UP), ((1, 1), LEFT), mixed_robots=True)
        env.robots[1].task, env.robots[0].task = give_retrieve(env, robot=0), None
        plan = [PICKUP, BACKWARD, BACKWARD, BACKWARD, TURN_RIGHT, TURN_RIGHT, DROPOFF]
        metrics = EpisodeMetrics()
        metrics.reset(env)
        while plan:
            # Off its turn the slow robot's action is ignored by the env, so any action will do.
            actions = [TURN_RIGHT, plan.pop(0) if env.ready(1) else TURN_RIGHT]
            metrics.observe(env, actions)
            env.step(actions)
            metrics.record(env)
        totals = metrics.totals()
        self.assertEqual(env.totals["deliveries"], 1)
        self.assertEqual((totals["job_steps"], totals["job_optimal"]), (7, 7))

    def test_a_job_replaced_in_the_same_step_is_counted_once_and_the_next_one_tracked(self):
        env = scripted(((1, 1), LEFT))
        give_retrieve(env)
        metrics = EpisodeMetrics()
        metrics.reset(env)
        for action in (PICKUP, BACKWARD, BACKWARD, BACKWARD, TURN_RIGHT, TURN_RIGHT):
            metrics.observe(env, [action])
            env.step([action])
            metrics.record(env)
        # A job waiting in the queue is handed out at the end of the delivering step. Built by
        # hand: the env would not draw one on the shelf the current job still reserves.
        env.queue.append(Task("retrieve", (0, 1), (5, 1)))
        metrics.observe(env, [DROPOFF])
        env.step([DROPOFF])
        metrics.record(env)
        self.assertIsNotNone(env.robots[0].task, "the next job arrived in the same step")
        self.assertEqual((metrics.totals()["job_steps"], metrics.totals()["job_optimal"]), (7, 7))
        self.assertIs(metrics.jobs[0][0], env.robots[0].task)

    def test_a_cell_nothing_can_face_has_no_optimum(self):
        env = scripted(((1, 1), LEFT))
        self.assertIsNone(job_optimum(env.grid, ((1, 1), LEFT), (0, 1), (99, 99)))

    def test_an_unfinished_job_is_not_counted(self):
        env = scripted(((1, 1), LEFT))
        give_retrieve(env)
        totals = play(env, [[PICKUP], [BACKWARD]])
        self.assertEqual((totals["job_steps"], totals["job_optimal"]), (0, 0))


class TestUndoneTurns(unittest.TestCase):
    def test_right_then_left_is_one_undone_turn(self):
        env = scripted(((2, 1), UP))
        give_retrieve(env)
        totals = play(env, [[TURN_RIGHT], [TURN_LEFT]])
        self.assertEqual((totals["undone_turns"], totals["acting_steps"]), (1, 2))
        self.assertEqual(totals["undone_turns_crowded"], 0, "nobody else is on the map")

    def test_two_turns_the_same_way_are_not_undone(self):
        env = scripted(((2, 1), UP))
        give_retrieve(env)
        self.assertEqual(play(env, [[TURN_RIGHT], [TURN_RIGHT]])["undone_turns"], 0)

    def test_an_action_in_between_breaks_the_pair(self):
        env = scripted(((2, 0), RIGHT))
        give_retrieve(env)
        self.assertEqual(play(env, [[TURN_RIGHT], [FORWARD], [TURN_LEFT]])["undone_turns"], 0)

    def test_an_undone_turn_next_to_a_robot_counts_as_crowded(self):
        # Robot 0's next shortest-path cell is (1, 1); robot 1 stands right next to it.
        env = scripted(((2, 1), UP), ((1, 0), DOWN))
        give_retrieve(env)
        totals = play(env, [[TURN_RIGHT, TURN_RIGHT], [TURN_LEFT, TURN_RIGHT]])
        self.assertEqual((totals["undone_turns"], totals["undone_turns_crowded"]), (1, 1))

    def test_a_robot_without_a_job_is_not_measured(self):
        env = scripted(((2, 1), UP))
        totals = play(env, [[TURN_RIGHT], [TURN_LEFT]])
        self.assertEqual((totals["undone_turns"], totals["acting_steps"]), (0, 0))


class TestNoWaitRate(unittest.TestCase):
    def test_there_is_no_wait_count_any_more(self):
        self.assertNotIn("waits", COUNTS)
        self.assertNotIn("wait_rate", rates([dict.fromkeys(COUNTS, 0)]))


class TestReversals(unittest.TestCase):
    def test_forward_then_backward_is_a_reversal(self):
        env = scripted(((2, 0), RIGHT))
        give_retrieve(env)
        self.assertEqual(play(env, [[FORWARD], [BACKWARD]])["reversals"], 1)

    def test_going_on_is_not_a_reversal(self):
        env = scripted(((1, 0), RIGHT))
        give_retrieve(env)
        self.assertEqual(play(env, [[FORWARD], [FORWARD]])["reversals"], 0)

    def test_backing_out_after_a_pickup_is_not_a_reversal(self):
        # The only way out of a dead-end pickup is the way in; the optimal job does exactly that.
        env = scripted(((2, 1), LEFT))
        give_retrieve(env)
        self.assertEqual(play(env, [[FORWARD], [PICKUP], [BACKWARD]])["reversals"], 0)

    def test_coming_back_after_turning_still_counts(self):
        env = scripted(((2, 0), RIGHT))
        give_retrieve(env)
        totals = play(env, [[FORWARD], [TURN_RIGHT], [TURN_RIGHT], [FORWARD]])
        self.assertEqual(totals["reversals"], 1)


class TestSlowRobots(unittest.TestCase):
    def test_an_ignored_off_step_action_is_not_counted(self):
        # mixed_robots: robot 1 is the slow one (period 2); at t=0 it acts only if (0 + 1) % 2 == 0.
        env = scripted(((1, 0), RIGHT), ((4, 2), LEFT), mixed_robots=True)
        give_retrieve(env, robot=0)
        env.robots[1].task = env.robots[0].task     # give the slow robot something to be measured on
        totals = play(env, [[TURN_RIGHT, TURN_RIGHT], [TURN_RIGHT, TURN_LEFT]])
        self.assertEqual(totals["acting_steps"], 3, "robot 0 twice, robot 1 only on its turn")


class TestRates(unittest.TestCase):
    def test_rates_pool_the_sums_rather_than_averaging_ratios(self):
        rows = [{"acting_steps": 10, "undone_turns": 1, "undone_turns_crowded": 1, "reversals": 0,
                 "job_steps": 30, "job_optimal": 10},
                {"acting_steps": 90, "undone_turns": 9, "undone_turns_crowded": 0, "reversals": 5,
                 "job_steps": 90, "job_optimal": 90}]
        out = rates(rows)
        self.assertAlmostEqual(out["job_ratio"], 120 / 100)
        self.assertAlmostEqual(out["undone_rate"], 10 / 100)
        self.assertAlmostEqual(out["crowded_share"], 1 / 10)
        self.assertAlmostEqual(out["reversal_rate"], 5 / 100)

    def test_no_data_gives_none_not_a_crash(self):
        out = rates([{"acting_steps": 0, "undone_turns": 0, "undone_turns_crowded": 0,
                      "reversals": 0, "job_steps": 0, "job_optimal": 0}])
        self.assertIsNone(out["job_ratio"])


if __name__ == "__main__":
    unittest.main()
