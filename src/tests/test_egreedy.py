import numpy as np
import unittest
from IPython import embed

from agents import EGreedy

class TestEGreedy(unittest.TestCase):

    def test_init(self):
        n_actions=100
        agent = EGreedy(n_actions)
        
        self.assertEqual(agent.round, 1)
        self.assertTrue(np.allclose(agent.means, -np.inf*np.ones(n_actions)))
        self.assertEqual(agent.action, None)
        self.assertTrue(np.allclose(agent.n_arm_pulls, np.zeros(n_actions)))
        self.assertEqual(agent.explore, None)
        self.assertEqual(agent.n_arms, n_actions)
        
        #check that initialition fails when the number of arms is incorrect
        with self.assertRaises(ValueError):
            EGreedy(n_arms=0)
        with self.assertRaises(ValueError):
            EGreedy(n_arms=1)
        
    def test_will_explore(self):
        n_iters = 10**5
        n_rounds = 10
        for r in range(1,n_rounds):
            n_explore = 0
            agent = EGreedy(n_arms=2)
            agent.round = r
            for j in range(1,n_iters):
                agent._will_explore()
                n_explore += agent.explore
            self.assertAlmostEqual(1/r, n_explore/n_iters, places=2)
        
    def test_sample_action_with_argument(self):
        n_actions = 2
        n_rounds=2
        for a in range(n_actions):
            vec = np.zeros(n_actions)
            agent = EGreedy(n_actions)
            for r in range(n_rounds):
                agent.sample_action(a)
                vec[a] += 1
                self.assertEqual(a, agent.action)
                self.assertEqual(agent.round, r+2)
                
                #check that the correct arm_pull is updated
                self.assertTrue(np.allclose(agent.n_arm_pulls, vec))
    
    #TODO
    def test_sample_action_without_argument(self):
        pass
    #TODO
    def test_update(self):
        n_actions=2
        n_rounds = 2
        for a in range(n_actions):
            for n in range(n_rounds):
                pass
        
if __name__ == '__main__':
    unittest.main()