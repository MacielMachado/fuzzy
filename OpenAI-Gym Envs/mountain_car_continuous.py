from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import numpy as np
import time
import gym

class MountainCarContinuousFuzzy():
    '''
    '''
    def __init__(self, random):
        '''
        '''
        self.random=random

    def run(self):
        '''
        '''
        self._create_env()
        self._game_loop()

    def _create_env(self):
        '''
        '''
        self.env = gym.make('MountainCarContinuous-v0')

    def _game_loop(self):
        '''
        '''
        self._create_fuzzy_agent()
        self.env.reset()
        action=[0]
        self.reward_list = []
        counter = 0
        while counter < 500:
            self.env.render()
            obs, rew, done, _ = self.env.step(action)
            # print(f'reward: {rew}n')
            self.agent.input['position']=obs[0]
            self.agent.input['velocity']=obs[1]
            if done: 
                self.env.reset()
                self.reward_list.append(rew)
                counter += 1
                print(counter)
            self.agent.compute()
            action = [self.agent.output['action']]
            if self.random:
                action = self.env.action_space.sample()
        self._plot_reward()
        self.env.reset()
        self.env.close()

    def _plot_reward(self):
        '''
        '''
        plt.plot(self.reward_list)
        plt.ylim([0, 120])
        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.title('MountainCarContinuous Rewards')
        plt.show()

    def _create_fuzzy_agent(self):
        '''
        '''
        self._fuzzy_antecedent()
        self._fuzzy_consequent()
        self._membership_functions()
        self._rules()
        self._control_system()

    def _fuzzy_antecedent(self):
        '''
        '''
        self.position_antecedent = ctrl.Antecedent(np.arange(-1.2, 0.6, 0.05),
                                                   'position')
        self.velocity_antecedent = ctrl.Antecedent(np.arange(-0.07, 0.07, 0.005),
                                                   'velocity')
    
    def _fuzzy_consequent(self):
        '''
        '''
        self.action_consequent = ctrl.Consequent(np.arange(-1, 1, 0.05),
                                                 'action')

    def _membership_functions(self):
        '''
        '''
        self.position_antecedent.automf(names=['baixa', 'media', 'alta'])
        self.velocity_antecedent.automf(names=['neg', 'baixa', 'pos'])
        self.action_consequent.automf(names=['neg', 'baixa', 'pos'])

        # Position
        self.position_antecedent['muito_baixa']=fuzz.trapmf(
                                            self.position_antecedent.universe,
                                            [-1.2, -1.2, -1.1, -1.0])

        self.position_antecedent['baixa']=fuzz.trimf(
                                            self.position_antecedent.universe,
                                            [-1.2, -1.0, -0.4])

        self.position_antecedent['media']=fuzz.trimf(
                                            self.position_antecedent.universe,
                                            [-0.6, -0.4, -0.3])

        self.position_antecedent['alta']=fuzz.trapmf(
                                            self.position_antecedent.universe,
                                            [-0.4, -0.3, 0.6, 0.6])

        # Velocity
        self.velocity_antecedent['neg']=fuzz.trapmf(
                                            self.velocity_antecedent.universe,
                                            [-0.07, -0.07, -0.01, 0])
        self.velocity_antecedent['baixa']=fuzz.trimf(
                                            self.velocity_antecedent.universe,
                                            [-0.01, -0, 0.01])
        self.velocity_antecedent['pos']=fuzz.trapmf(
                                            self.velocity_antecedent.universe,
                                            [0, 0.01, 0.07, 0.07])

        # Action
        self.action_consequent['neg']=fuzz.trapmf(
                                            self.action_consequent.universe, 
                                            [-1, -1, -0.1, 0])
        self.action_consequent['baixa']=fuzz.trimf(
                                            self.action_consequent.universe, 
                                            [-0.1, 0, 0.1])
        self.action_consequent['pos']=fuzz.trapmf(
                                            self.action_consequent.universe, 
                                            [0, 0.1, 1, 1])

    def _rules(self):
        '''
        '''
        rule1 = ctrl.Rule(self.position_antecedent['muito_baixa'] & 
                          self.velocity_antecedent['neg'],
                          self.action_consequent['baixa'])
        rule2 = ctrl.Rule(self.position_antecedent['muito_baixa'] & 
                          self.velocity_antecedent['baixa'],
                          self.action_consequent['baixa'])
        rule3 = ctrl.Rule(self.position_antecedent['muito_baixa'] & 
                          self.velocity_antecedent['pos'],
                          self.action_consequent['baixa'])


        rule4 = ctrl.Rule(self.position_antecedent['baixa'] & 
                          self.velocity_antecedent['neg'],
                          self.action_consequent['neg'])
        rule5 = ctrl.Rule(self.position_antecedent['baixa'] & 
                          self.velocity_antecedent['baixa'],
                          self.action_consequent['pos'])
        rule6 = ctrl.Rule(self.position_antecedent['baixa'] & 
                          self.velocity_antecedent['pos'],
                          self.action_consequent['pos'])


        rule7 = ctrl.Rule(self.position_antecedent['media'] & 
                          self.velocity_antecedent['neg'],
                          self.action_consequent['neg'])
        rule8 = ctrl.Rule(self.position_antecedent['media'] & 
                          self.velocity_antecedent['baixa'],
                          self.action_consequent['pos'])
        rule9 = ctrl.Rule(self.position_antecedent['media'] & 
                          self.velocity_antecedent['pos'],
                          self.action_consequent['pos'])


        rule10 = ctrl.Rule(self.position_antecedent['alta'] & 
                          self.velocity_antecedent['neg'],
                          self.action_consequent['neg'])
        rule11 = ctrl.Rule(self.position_antecedent['alta'] & 
                          self.velocity_antecedent['baixa'],
                          self.action_consequent['neg'])
        rule12 = ctrl.Rule(self.position_antecedent['alta'] & 
                          self.velocity_antecedent['pos'],
                          self.action_consequent['pos'])
        
        self.rules = [rule1, rule2, rule3, rule4, rule5, rule6,
                      rule7, rule8, rule9, rule10, rule11, rule12]

    def _control_system(self):
        '''
        '''
        ctrl_system=ctrl.ControlSystem(self.rules)
        self.agent = ctrl.ControlSystemSimulation(ctrl_system)

if __name__ == '__main__':
    MountainCarContinuousFuzzy(random=False).run()