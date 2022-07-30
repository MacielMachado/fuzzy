from skfuzzy import control as ctrl
import skfuzzy as fuzz
import numpy as np
import mujoco
import time
import gym

class InvertedPendulum():
    '''
    '''
    def __init__(self):
        '''
        '''
        pass

    def run(self):
        '''
        '''
        self._create_env()
        self._game_loop()

    def _create_env(self):
        '''
        '''
        self.env = gym.make('Pendulum-v1', g=16.44)

    def _game_loop(self):
        '''
        '''
        self._create_fuzzy_agent()
        self.env.reset()
        action=[0]
        while True:
            # time.sleep(1)
            self.env.render()
            action = self.env.action_space.sample()
            action=[2]
            obs, rew, done, _ = self.env.step(action)

            angle = 180*np.arctan(obs[1]/obs[0])/np.pi + 90 - 360

            if angle > 360:
                angle = angle - 360
            elif angle < 0:
                angle = angle + 360
            self.agent.input['angles']=angle
            self.agent.input['velocity']=obs[2]
            if done: break
            self.agent.compute()
            action = [self.agent.output['torque']]
            print(f'angle: {angle}, velocity: {obs[2]}, action: {action}')
        self.env.reset()
        self.env.close()

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
        self.angle_position_antecedent = ctrl.Antecedent(np.arange(0, 360, 15), 'angles')
        self.velocity_antecedent = ctrl.Antecedent(np.arange(-8, 8, 1), 'velocity')
    
    def _fuzzy_consequent(self):
        '''
        '''
        self.torque_consequent = ctrl.Consequent(np.arange(-2, 2, 0.1), 'torque')

    def _membership_functions(self):
        '''
        '''
        self.angle_position_antecedent.automf(names=['um', 'dois', 'tres', 'quatro',
                                        'cinco', 'seis', 'sete', 'oito'])
        self.velocity_antecedent.automf(names=['muito_pos', 'pouco_pos', 'quase_zero',
                                        'pouco_neg', 'muito_neg'])
        self.torque_consequent.automf(names=['forte_pos', 'medio_pos', 'fraco_pos', 'fraquissimo_pos',
                                        'fraquissimo_neg', 'fraco_neg', 'medio_neg', 'forte_neg'])

        # Angle
        self.angle_position_antecedent['um'] = fuzz.trimf(self.angle_position_antecedent.universe,
                                                    [0, 30, 45])
        self.angle_position_antecedent['dois'] = fuzz.trimf(self.angle_position_antecedent.universe,
                                                    [45, 75, 100])
        self.angle_position_antecedent['tres'] = fuzz.trimf(self.angle_position_antecedent.universe,
                                                    [80, 110, 135])
        self.angle_position_antecedent['quatro'] = fuzz.trimf(self.angle_position_antecedent.universe,
                                                    [135, 150, 180])
        self.angle_position_antecedent['cinco'] = fuzz.trimf(self.angle_position_antecedent.universe,
                                                    [180, 195, 225])
        self.angle_position_antecedent['seis'] = fuzz.trimf(self.angle_position_antecedent.universe,
                                                    [225, 240, 270])
        self.angle_position_antecedent['sete'] = fuzz.trimf(self.angle_position_antecedent.universe,
                                                    [270, 285, 315])
        self.angle_position_antecedent['oito'] = fuzz.trimf(self.angle_position_antecedent.universe,
                                                    [315, 330, 360])

        # Velocity
        self.velocity_antecedent['muito_neg'] = fuzz.trapmf(self.velocity_antecedent.universe,
                                                    [-8, -8, -6, -5])
        self.velocity_antecedent['pouco_neg'] = fuzz.trimf(self.velocity_antecedent.universe,
                                                    [-6, -4, -1])
        self.velocity_antecedent['quase_zero'] = fuzz.trimf(self.velocity_antecedent.universe,
                                                    [-2, 0, 2])
        self.velocity_antecedent['pouco_pos'] = fuzz.trimf(self.velocity_antecedent.universe,
                                                    [1, 4, 6])
        self.velocity_antecedent['muito_pos'] = fuzz.trapmf(self.velocity_antecedent.universe,
                                                    [5, 6, 8, 8])

        # Torque
        self.torque_consequent['forte_neg'] = fuzz.trapmf(self.torque_consequent.universe,
                                                    [-2, -2, -1.5, -1])
        self.torque_consequent['medio_neg'] = fuzz.trimf(self.torque_consequent.universe,
                                                    [-1.1, -0.80, -0.5])
        self.torque_consequent['fraco_neg'] = fuzz.trimf(self.torque_consequent.universe,
                                                    [-0.55, -0.3, -0.2])
        self.torque_consequent['fraquissimo_neg'] = fuzz.trimf(self.torque_consequent.universe,
                                                    [-0.2, -0.05, 0])
        self.torque_consequent['fraquissimo_pos'] = fuzz.trimf(self.torque_consequent.universe,
                                                    [0, 0.05, 0.2])
        self.torque_consequent['fraco_pos'] = fuzz.trimf(self.torque_consequent.universe,
                                                    [0.2, 0.3, 0.55])
        self.torque_consequent['medio_pos'] = fuzz.trimf(self.torque_consequent.universe,
                                                    [0.5, 0.8, 1.1])
        self.torque_consequent['forte_pos'] = fuzz.trapmf(self.torque_consequent.universe,
                                                    [1, 1.5, 2, 2])
        
    def _rules(self):
        '''
        '''
        rule1_1 = ctrl.Rule(self.angle_position_antecedent['um'] &
                            self.velocity_antecedent['muito_pos'],
                            self.torque_consequent['fraco_pos'])
        rule1_2 = ctrl.Rule(self.angle_position_antecedent['um'] &
                            self.velocity_antecedent['pouco_pos'],
                            self.torque_consequent['fraco_pos'])
        rule1_3 = ctrl.Rule(self.angle_position_antecedent['um'] &
                            self.velocity_antecedent['quase_zero'],
                            self.torque_consequent['fraquissimo_pos'])
        rule1_4 = ctrl.Rule(self.angle_position_antecedent['um'] &
                            self.velocity_antecedent['pouco_neg'],
                            self.torque_consequent['forte_pos'])
        rule1_5 = ctrl.Rule(self.angle_position_antecedent['um'] &
                            self.velocity_antecedent['muito_neg'],
                            self.torque_consequent['forte_pos'])



        rule2_1 = ctrl.Rule(self.angle_position_antecedent['dois'] &
                            self.velocity_antecedent['muito_pos'],
                            self.torque_consequent['forte_neg'])
        rule2_2 = ctrl.Rule(self.angle_position_antecedent['dois'] &
                            self.velocity_antecedent['pouco_pos'],
                            self.torque_consequent['forte_neg'])
        rule2_3 = ctrl.Rule(self.angle_position_antecedent['dois'] &
                            self.velocity_antecedent['quase_zero'],
                            self.torque_consequent['forte_neg'])
        rule2_4 = ctrl.Rule(self.angle_position_antecedent['dois'] &
                            self.velocity_antecedent['pouco_neg'],
                            self.torque_consequent['forte_neg'])
        rule2_5 = ctrl.Rule(self.angle_position_antecedent['dois'] &
                            self.velocity_antecedent['muito_neg'],
                            self.torque_consequent['forte_neg'])



        rule3_1 = ctrl.Rule(self.angle_position_antecedent['tres'] &
                            self.velocity_antecedent['muito_pos'],
                            self.torque_consequent['forte_neg'])
        rule3_2 = ctrl.Rule(self.angle_position_antecedent['tres'] &
                            self.velocity_antecedent['pouco_pos'],
                            self.torque_consequent['forte_neg'])
        rule3_3 = ctrl.Rule(self.angle_position_antecedent['tres'] &
                            self.velocity_antecedent['quase_zero'],
                            self.torque_consequent['forte_neg'])
        rule3_4 = ctrl.Rule(self.angle_position_antecedent['tres'] &
                            self.velocity_antecedent['pouco_neg'],
                            self.torque_consequent['forte_neg'])
        rule3_5 = ctrl.Rule(self.angle_position_antecedent['tres'] &
                            self.velocity_antecedent['muito_neg'],
                            self.torque_consequent['forte_neg'])



        rule4_1 = ctrl.Rule(self.angle_position_antecedent['quatro'] &
                            self.velocity_antecedent['muito_pos'],
                            self.torque_consequent['forte_neg'])
        rule4_2 = ctrl.Rule(self.angle_position_antecedent['quatro'] &
                            self.velocity_antecedent['pouco_pos'],
                            self.torque_consequent['forte_neg'])
        rule4_3 = ctrl.Rule(self.angle_position_antecedent['quatro'] &
                            self.velocity_antecedent['quase_zero'],
                            self.torque_consequent['forte_neg'])
        rule4_4 = ctrl.Rule(self.angle_position_antecedent['quatro'] &
                            self.velocity_antecedent['pouco_neg'],
                            self.torque_consequent['forte_neg'])
        rule4_5 = ctrl.Rule(self.angle_position_antecedent['quatro'] &
                            self.velocity_antecedent['muito_neg'],
                            self.torque_consequent['fraquissimo_pos'])



        rule5_1 = ctrl.Rule(self.angle_position_antecedent['cinco'] &
                            self.velocity_antecedent['muito_pos'],
                            self.torque_consequent['fraco_pos'])
        rule5_2 = ctrl.Rule(self.angle_position_antecedent['cinco'] &
                            self.velocity_antecedent['pouco_pos'],
                            self.torque_consequent['medio_pos'])
        rule5_3 = ctrl.Rule(self.angle_position_antecedent['cinco'] &
                            self.velocity_antecedent['quase_zero'],
                            self.torque_consequent['forte_pos'])
        rule5_4 = ctrl.Rule(self.angle_position_antecedent['cinco'] &
                            self.velocity_antecedent['pouco_neg'],
                            self.torque_consequent['forte_pos'])
        rule5_5 = ctrl.Rule(self.angle_position_antecedent['cinco'] &
                            self.velocity_antecedent['muito_neg'],
                            self.torque_consequent['medio_neg'])



        rule6_1 = ctrl.Rule(self.angle_position_antecedent['seis'] &
                            self.velocity_antecedent['muito_pos'],
                            self.torque_consequent['medio_pos'])
        rule6_2 = ctrl.Rule(self.angle_position_antecedent['seis'] &
                            self.velocity_antecedent['pouco_pos'],
                            self.torque_consequent['forte_pos'])
        rule6_3 = ctrl.Rule(self.angle_position_antecedent['seis'] &
                            self.velocity_antecedent['quase_zero'],
                            self.torque_consequent['forte_pos'])
        rule6_4 = ctrl.Rule(self.angle_position_antecedent['seis'] &
                            self.velocity_antecedent['pouco_neg'],
                            self.torque_consequent['forte_pos'])
        rule6_5 = ctrl.Rule(self.angle_position_antecedent['seis'] &
                            self.velocity_antecedent['muito_neg'],
                            self.torque_consequent['medio_neg'])



        rule7_1 = ctrl.Rule(self.angle_position_antecedent['sete'] &
                            self.velocity_antecedent['muito_pos'],
                            self.torque_consequent['medio_pos'])
        rule7_2 = ctrl.Rule(self.angle_position_antecedent['sete'] &
                            self.velocity_antecedent['pouco_pos'],
                            self.torque_consequent['forte_pos'])
        rule7_3 = ctrl.Rule(self.angle_position_antecedent['sete'] &
                            self.velocity_antecedent['quase_zero'],
                            self.torque_consequent['forte_neg'])
        rule7_4 = ctrl.Rule(self.angle_position_antecedent['sete'] &
                            self.velocity_antecedent['pouco_neg'],
                            self.torque_consequent['medio_neg'])
        rule7_5 = ctrl.Rule(self.angle_position_antecedent['sete'] &
                            self.velocity_antecedent['muito_neg'],
                            self.torque_consequent['fraco_neg'])



        rule8_1 = ctrl.Rule(self.angle_position_antecedent['oito'] &
                            self.velocity_antecedent['muito_pos'],
                            self.torque_consequent['medio_pos'])
        rule8_2 = ctrl.Rule(self.angle_position_antecedent['oito'] &
                            self.velocity_antecedent['pouco_pos'],
                            self.torque_consequent['forte_pos'])
        rule8_3 = ctrl.Rule(self.angle_position_antecedent['oito'] &
                            self.velocity_antecedent['quase_zero'],
                            self.torque_consequent['forte_neg'])
        rule8_4 = ctrl.Rule(self.angle_position_antecedent['oito'] &
                            self.velocity_antecedent['pouco_neg'],
                            self.torque_consequent['medio_neg'])
        rule8_5 = ctrl.Rule(self.angle_position_antecedent['oito'] &
                            self.velocity_antecedent['muito_neg'],
                            self.torque_consequent['fraco_neg'])









        # rule1_1 = ctrl.Rule(self.angle_position_antecedent['um'] &
        #                     self.velocity_antecedent['muito_pos'],
        #                     self.torque_consequent['fraquissimo_neg'])
        # rule1_2 = ctrl.Rule(self.angle_position_antecedent['um'] &
        #                     self.velocity_antecedent['pouco_pos'],
        #                     self.torque_consequent['fraquissimo_pos'])
        # rule1_3 = ctrl.Rule(self.angle_position_antecedent['um'] &
        #                     self.velocity_antecedent['quase_zero'],
        #                     self.torque_consequent['fraquissimo_pos'])
        # rule1_4 = ctrl.Rule(self.angle_position_antecedent['um'] &
        #                     self.velocity_antecedent['pouco_neg'],
        #                     self.torque_consequent['fraquissimo_pos'])
        # rule1_5 = ctrl.Rule(self.angle_position_antecedent['um'] &
        #                     self.velocity_antecedent['muito_neg'],
        #                     self.torque_consequent['fraquissimo_pos'])



        # rule2_1 = ctrl.Rule(self.angle_position_antecedent['dois'] &
        #                     self.velocity_antecedent['muito_pos'],
        #                     self.torque_consequent['fraquissimo_neg'])
        # rule2_2 = ctrl.Rule(self.angle_position_antecedent['dois'] &
        #                     self.velocity_antecedent['pouco_pos'],
        #                     self.torque_consequent['fraquissimo_pos'])
        # rule2_3 = ctrl.Rule(self.angle_position_antecedent['dois'] &
        #                     self.velocity_antecedent['quase_zero'],
        #                     self.torque_consequent['fraquissimo_pos'])
        # rule2_4 = ctrl.Rule(self.angle_position_antecedent['dois'] &
        #                     self.velocity_antecedent['pouco_neg'],
        #                     self.torque_consequent['fraquissimo_pos'])
        # rule2_5 = ctrl.Rule(self.angle_position_antecedent['dois'] &
        #                     self.velocity_antecedent['muito_neg'],
        #                     self.torque_consequent['fraquissimo_pos'])



        # rule3_1 = ctrl.Rule(self.angle_position_antecedent['tres'] &
        #                     self.velocity_antecedent['muito_pos'],
        #                     self.torque_consequent['fraquissimo_neg'])
        # rule3_2 = ctrl.Rule(self.angle_position_antecedent['tres'] &
        #                     self.velocity_antecedent['pouco_pos'],
        #                     self.torque_consequent['fraquissimo_neg'])
        # rule3_3 = ctrl.Rule(self.angle_position_antecedent['tres'] &
        #                     self.velocity_antecedent['quase_zero'],
        #                     self.torque_consequent['fraquissimo_neg'])
        # rule3_4 = ctrl.Rule(self.angle_position_antecedent['tres'] &
        #                     self.velocity_antecedent['pouco_neg'],
        #                     self.torque_consequent['fraquissimo_neg'])
        # rule3_5 = ctrl.Rule(self.angle_position_antecedent['tres'] &
        #                     self.velocity_antecedent['muito_neg'],
        #                     self.torque_consequent['fraquissimo_pos'])



        # rule4_1 = ctrl.Rule(self.angle_position_antecedent['quatro'] &
        #                     self.velocity_antecedent['muito_pos'],
        #                     self.torque_consequent['fraquissimo_neg'])
        # rule4_2 = ctrl.Rule(self.angle_position_antecedent['quatro'] &
        #                     self.velocity_antecedent['pouco_pos'],
        #                     self.torque_consequent['fraquissimo_neg'])
        # rule4_3 = ctrl.Rule(self.angle_position_antecedent['quatro'] &
        #                     self.velocity_antecedent['quase_zero'],
        #                     self.torque_consequent['fraquissimo_neg'])
        # rule4_4 = ctrl.Rule(self.angle_position_antecedent['quatro'] &
        #                     self.velocity_antecedent['pouco_neg'],
        #                     self.torque_consequent['fraquissimo_neg'])
        # rule4_5 = ctrl.Rule(self.angle_position_antecedent['quatro'] &
        #                     self.velocity_antecedent['muito_neg'],
        #                     self.torque_consequent['fraquissimo_pos'])



        # rule5_1 = ctrl.Rule(self.angle_position_antecedent['cinco'] &
        #                     self.velocity_antecedent['muito_pos'],
        #                     self.torque_consequent['fraquissimo_pos'])
        # rule5_2 = ctrl.Rule(self.angle_position_antecedent['cinco'] &
        #                     self.velocity_antecedent['pouco_pos'],
        #                     self.torque_consequent['fraquissimo_pos'])
        # rule5_3 = ctrl.Rule(self.angle_position_antecedent['cinco'] &
        #                     self.velocity_antecedent['quase_zero'],
        #                     self.torque_consequent['fraquissimo_pos'])
        # rule5_4 = ctrl.Rule(self.angle_position_antecedent['cinco'] &
        #                     self.velocity_antecedent['pouco_neg'],
        #                     self.torque_consequent['fraquissimo_pos'])
        # rule5_5 = ctrl.Rule(self.angle_position_antecedent['cinco'] &
        #                     self.velocity_antecedent['muito_neg'],
        #                     self.torque_consequent['fraquissimo_neg'])



        # rule6_1 = ctrl.Rule(self.angle_position_antecedent['seis'] &
        #                     self.velocity_antecedent['muito_pos'],
        #                     self.torque_consequent['fraquissimo_pos'])
        # rule6_2 = ctrl.Rule(self.angle_position_antecedent['seis'] &
        #                     self.velocity_antecedent['pouco_pos'],
        #                     self.torque_consequent['fraquissimo_pos'])
        # rule6_3 = ctrl.Rule(self.angle_position_antecedent['seis'] &
        #                     self.velocity_antecedent['quase_zero'],
        #                     self.torque_consequent['fraquissimo_pos'])
        # rule6_4 = ctrl.Rule(self.angle_position_antecedent['seis'] &
        #                     self.velocity_antecedent['pouco_neg'],
        #                     self.torque_consequent['fraquissimo_pos'])
        # rule6_5 = ctrl.Rule(self.angle_position_antecedent['seis'] &
        #                     self.velocity_antecedent['muito_neg'],
        #                     self.torque_consequent['fraquissimo_neg'])



        # rule7_1 = ctrl.Rule(self.angle_position_antecedent['sete'] &
        #                     self.velocity_antecedent['muito_pos'],
        #                     self.torque_consequent['fraquissimo_neg'])
        # rule7_2 = ctrl.Rule(self.angle_position_antecedent['sete'] &
        #                     self.velocity_antecedent['pouco_pos'],
        #                     self.torque_consequent['fraquissimo_neg'])
        # rule7_3 = ctrl.Rule(self.angle_position_antecedent['sete'] &
        #                     self.velocity_antecedent['quase_zero'],
        #                     self.torque_consequent['fraquissimo_neg'])
        # rule7_4 = ctrl.Rule(self.angle_position_antecedent['sete'] &
        #                     self.velocity_antecedent['pouco_neg'],
        #                     self.torque_consequent['fraquissimo_neg'])
        # rule7_5 = ctrl.Rule(self.angle_position_antecedent['sete'] &
        #                     self.velocity_antecedent['muito_neg'],
        #                     self.torque_consequent['fraquissimo_neg'])



        # rule8_1 = ctrl.Rule(self.angle_position_antecedent['oito'] &
        #                     self.velocity_antecedent['muito_pos'],
        #                     self.torque_consequent['fraquissimo_neg'])
        # rule8_2 = ctrl.Rule(self.angle_position_antecedent['oito'] &
        #                     self.velocity_antecedent['pouco_pos'],
        #                     self.torque_consequent['fraquissimo_neg'])
        # rule8_3 = ctrl.Rule(self.angle_position_antecedent['oito'] &
        #                     self.velocity_antecedent['quase_zero'],
        #                     self.torque_consequent['fraquissimo_neg'])
        # rule8_4 = ctrl.Rule(self.angle_position_antecedent['oito'] &
        #                     self.velocity_antecedent['pouco_neg'],
        #                     self.torque_consequent['fraquissimo_neg'])
        # rule8_5 = ctrl.Rule(self.angle_position_antecedent['oito'] &
        #                     self.velocity_antecedent['muito_neg'],
        #                     self.torque_consequent['fraquissimo_neg'])











        self.rules = [rule1_1, rule1_2, rule1_3, rule1_4, rule1_5,
                      rule2_1, rule2_2, rule2_3, rule2_4, rule2_5,
                      rule3_1, rule3_2, rule3_3, rule3_4, rule3_5,
                      rule4_1, rule4_2, rule4_3, rule4_4, rule4_5,
                      rule5_1, rule5_2, rule5_3, rule5_4, rule5_5,
                      rule6_1, rule6_2, rule6_3, rule6_4, rule6_5,
                      rule7_1, rule7_2, rule7_3, rule7_4, rule7_5,
                      rule8_1, rule8_2, rule8_3, rule8_4, rule8_5,
                      ]

    def _control_system(self):
        '''
        '''
        ctrl_system=ctrl.ControlSystem(self.rules)
        self.agent = ctrl.ControlSystemSimulation(ctrl_system)



if __name__ == '__main__':
    InvertedPendulum().run()