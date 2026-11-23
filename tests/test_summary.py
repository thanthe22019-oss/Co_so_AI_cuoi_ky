import unittest
from experiments.summarize import aggregate, findings


class SummaryTests(unittest.TestCase):
    def test_reflex_without_seed_zero(self):
        rows = [dict(size='15', agent='reflex', seed=str(seed), phase='eval',
                     score='12', food_fraction='.5', steps='10', outcome='timeout',
                     decision_seconds='.1', model_entries='0', train_seconds='0')
                for seed in [1, 2]]
        seeds, summary = aggregate(rows)
        self.assertEqual(len(seeds), 1)
        self.assertEqual(summary[0]['games'], 1)

    def test_findings_accept_agent_subset(self):
        text = findings([dict(size=15, agent='tabular', food_fraction=.25)])
        self.assertIn('tabular', text)
        self.assertNotIn('so với', text)


if __name__ == '__main__':
    unittest.main()
