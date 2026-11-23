import copy
import unittest
from experiments import benchmark as b


class ExperimentTests(unittest.TestCase):
    def test_maps_are_reproducible_and_connected(self):
        for size in (15, 25, 35):
            rows = b.make_layout(size, 42)
            self.assertEqual(rows, b.make_layout(size, 42))
            self.assertEqual((len(rows), len(rows[0])), (size, size))
            b.validate_layout(rows)

    def test_unreachable_food_rejected(self):
        with self.assertRaises(ValueError):
            b.validate_layout(['%%%%%%%', '%P% .G%', '%%%%%%%'])

    def test_no_food_rejected(self):
        with self.assertRaises(ValueError):
            b.validate_layout(['%%%%%', '%P G%', '%%%%%'])

    def test_step_limit_and_evaluation_freeze(self):
        layout = b.Layout(b.make_layout(15, 42))
        for kind in ('tabular', 'simple', 'improved', 'no-danger'):
            agent = b.make_agent(kind)
            b.episode(agent, layout, 42, 10, train=True)
            snapshot = copy.deepcopy((dict(agent.qValues), dict(getattr(agent, 'weights', {}))))
            result = b.episode(agent, layout, 43, 1, train=False)
            self.assertEqual(result['steps'], 1)
            self.assertEqual(result['outcome'], 'timeout')
            self.assertEqual(snapshot, (dict(agent.qValues), dict(getattr(agent, 'weights', {}))))

    def test_resume_only_accepts_complete_evaluation_and_budget(self):
        rows = [dict(size='15',agent='reflex',seed='0',phase='eval',training_steps='0') for _ in range(2)]
        rows += [dict(size='15',agent='simple',seed='0',phase='eval',training_steps='500') for _ in range(2)]
        rows += [dict(size='25',agent='simple',seed='1',phase='eval',training_steps='1000')]
        self.assertEqual(b.completed_runs(rows, 2, 1000), {(15,'reflex',0)})


if __name__ == '__main__':
    unittest.main()
