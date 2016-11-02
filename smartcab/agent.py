import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import math  

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here

        self.Q = {}           # Pushpendra_pratap has declared and initialized it -> Q(s,a) and the following alpha & gamma
        # self.alpha = 0.3      # Learning rate, chose between 0 <= alpha <= 1
        # self.gamma = 0.7      # Discount factor, chose between 0 <= gamma <= 1
        # self.alpha, self.gamma = self.env.alpha_gamma(self)
        self.epsilon = 0.0    # Exploration rate, stictly chose between 0 < epsilon < 1 , i.e. how much we want to Explore

        self.successes = []
        self.total_deadline_for_one_trial = [False, 0]       # used with epsilon
        self.random_action = 0                               # for Exploration
        self.argmax_action = 0                               # for Exploitation
        self.total_num_of_trials = 1                         # used in updating self.epsilon

        self.d = {}                                          # used in self.env.compute_dist(arg1, arg2)

        self.optimal_path_found = []
        self.reward_status_for_one_trial = True    # for any -ve reward gain in any iteration of one trial, it will become False 

        # action = argmax Q(s,a) ; with probability (1-epsilon)
        #        = random action ; with probability (epsilon)

        all_actions = self.env.valid_actions   # all possible action an agent can take i.e. [None, 'forward', 'left', 'right']
        all_actions_except_None = ['forward', 'left', 'right']
        traffic_light = ['red','green']        # or, traffic_light = self.env.TrafficLight.valid_states
        oncoming, left, right, waypoint = all_actions, all_actions, all_actions, all_actions_except_None

        for i in traffic_light:
            for j in oncoming:
                for k in left:
                    for l in right:    
                        for m in waypoint:
                            self.Q[(i,j,k,l,m)] = {None:0, 'forward':0, 'left':0, 'right':0}  
                            # self.Q[(i,j,k,l,m)] = {None:1.5, 'forward':1.5, 'left':1.5, 'right':1.5} 
        # -----------------------------------------------------------------------
        # or, we can initialize the Q-table like the following :

        # from collections import defaultdict
        # Q = defaultdict(lambda: {None:0, 'forward':0, 'left':0, 'right':0})

        # Now any key will have the value {None:0, 'forward':0, 'left':0, 'right':0} by default,
        # even if the dict has never seen that key before. 
        # -------------------------------------------------------------------------
 
    def reset(self, destination=None):
        self.planner.route_to(destination)

        # TODO: Prepare for a new trip; reset any variables here, if required
        self.successes.append(False)
        self.total_deadline_for_one_trial = [False, 0]        # used with epsilon
        self.random_action = 0
        self.argmax_action = 0
        # self.epsilon = 1/(self.total_num_of_trials)                 # similar to f(x)=1/x
        # self.epsilon = 1/(math.sqrt(self.total_num_of_trials))      # it's equal to math.pow(self.total_num_of_trials,1/2.0)
        # self.epsilon = 1/math.pow(self.total_num_of_trials,1/1.1)
        self.epsilon = 1/math.pow(self.total_num_of_trials,1/self.used_in_epsilon)
        self.total_num_of_trials += 1

        self.d['destination'] = [0, False]
        self.d['location'] = [0, False]

        self.optimal_path_found.append(-1)
        self.reward_status_for_one_trial = True


    def update(self, t):
        
        # Gather inputs
        
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
                                                           # it will return either 'forward' or 'left' or 'right' or None
                                                           
        inputs = self.env.sense(self)                      # it will return {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}
        
        deadline = self.env.get_deadline(self)

        if (self.total_deadline_for_one_trial[0] == False):   # once for every Trial, just in the beginning of Trial - Pushpendra
            self.total_deadline_for_one_trial[0] = True
            self.total_deadline_for_one_trial[1] = deadline

            self.d['location'][0] = self.env.agent_states[self]['location']
            self.d['location'][1] = True

            self.d['destination'][0] = self.env.agent_states[self]['destination']
            self.d['destination'][1] = True

            print 'd = {}'.format(self.d)

        start_coordinate = self.d['location'][0]
        end_coordinate = self.d['destination'][0]


        # TODO: Update state

        # self.state = (inputs, self.next_waypoint, deadline)   # Pushpendra_pratap have written this line, initially it was blank space
        if (self.next_waypoint != None):
            self.state = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)  # For Question 2

        # TODO: Select action according to your policy

        # action = None
        # action = random.choice([None, 'forward', 'left', 'right'])    # For Question 1
        # action = max(self.Q[self.state], key=self.Q[self.state].get)  # initially, it will not be much diff. than random choice.
        #                                                                  Do some Exploitation - Exploration for action variable   
        # -----------------------------------------------------------------------------------------------
        if (self.total_deadline_for_one_trial[0] == True):

            total_samples_for_random_action = self.total_deadline_for_one_trial[1] * self.epsilon
            diff = total_samples_for_random_action - int(total_samples_for_random_action)
            if (diff >= 0.5):
                total_samples_for_random_action = int(total_samples_for_random_action) + 1
            else:
                total_samples_for_random_action = int(total_samples_for_random_action)
            total_samples_for_argmax_action = self.total_deadline_for_one_trial[1] - total_samples_for_random_action

            # temp = random.randint(1,2)                                                   # METHOD 1
            # if (temp == 1):
            #     self.random_action = self.random_action + 1
            #     if (self.random_action >= total_samples_for_random_action):
            #         action = max(self.Q[self.state], key=self.Q[self.state].get)
            #     else:
            #         action = random.choice([None, 'forward', 'left', 'right'])
            # else:
            #     self.argmax_action = self.argmax_action + 1
            #     if (self.argmax_action >= total_samples_for_argmax_action):
            #         action = random.choice([None, 'forward', 'left', 'right'])
            #     else:
            #         action = max(self.Q[self.state], key=self.Q[self.state].get)
                
            if (self.random_action >= total_samples_for_random_action):                    # METHOD 2 - it's better than METHOD 1
                action = max(self.Q[self.state], key=self.Q[self.state].get)
                self.argmax_action = self.argmax_action + 1
            else:
                # action = random.choice([None, 'forward', 'left', 'right'])
                action = random.choice(['forward', 'left', 'right'])
                self.random_action = self.random_action + 1

        # ----------------------------------------------------------------------------------------------------

        # present_location = self.env.agent_states[self]['location']     # Pushpendra, both these lines are for Debugging purpose
        # print 'present_location = {}'.format(present_location) 

        # Execute action and get reward
        reward = self.env.act(self, action) # Once we do self.env.act(self, action) in agent.py we are moving our agent to another state .
                                            # when it calls self.env.act() it's position can be changed so that it lands in a different intersection .

        # ---------------------------------------------------------------------------------------------
        # To find how many times agent is able to find optimal path for a destination

        if (reward < 0):
            self.reward_status_for_one_trial = False

        if (self.env.done == True):
            # self.total_deadline_for_one_trial[1]/5.0
            # self.env.compute_dist(start_coordinate, end_coordinate)
            # here, we divided by 5.0 because in self.env.reset(), initialization is like, deadline = self.env.compute_dist(start, destination) * 5
            # print 'compute_dist() = {}'.format(self.env.compute_dist(start_coordinate, end_coordinate))
            if (((self.argmax_action + self.random_action) <= self.env.compute_dist(start_coordinate, end_coordinate)) and    \
                                                       self.reward_status_for_one_trial == True):
                self.optimal_path_found[-1] = 1
            else:
                self.optimal_path_found[-1] = 0

        #  In self.optimal_path_found[] , the following values means :-
        # -1 = smartcab failed to reach the destination
        #  0 = smartcab reaches to the destination but not by following optimal path
        #  1 = smartcab reaches to the destination using optimal path 

        # -----------------------------------------------------------------------------------------------

        # Pushpendra_pratap calling again for Q(s',a')
        next_inputs = self.env.sense(self)                      # To get our s'
        self.next_next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator

        if (self.next_next_waypoint != None):
            self.next_state = (next_inputs['light'], next_inputs['oncoming'], next_inputs['left'], next_inputs['right'],  \
                               self.next_next_waypoint)

        # TODO: Learn policy based on state, action, reward 
        # self.Q[self.state][action] = reward + self.gamma*max(self.Q[self.next_state].values())         # For Question 3
            self.Q[self.state][action] = ( (1-self.alpha)*self.Q[self.state][action] +  \
                               self.alpha*(reward + self.gamma*max(self.Q[self.next_state].values())) )    # For Question 4 & 5       


        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, alpha = {}, gamma = {}".format(deadline,   \
                                                                   inputs, action, reward, self.alpha, self.gamma)       # [debug]

        # print 'compute_dist() = {}'.format(self.env.compute_dist(start_coordinate, end_coordinate))
       # Pushpendra did this for model evaluation purpose
        my_location = self.env.agent_states[self]['location']
        my_destination = self.env.agent_states[self]['destination']

        if (my_location == my_destination):       # Just for measuring performance, corresponding to all the Trials
            self.successes[-1] = True

   
        # next_location = self.env.agent_states[self]['location']  # Pushpendra, both these lines are for Debugging purpose
        # print 'next_location = {}'.format(next_location) 
        # print 'my_location = {}, my_destination = {}'.format(my_location, my_destination)       



def run(i, j, k):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
    
    # Now simulate it
    sim = Simulator(e, update_delay=0.00000005, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    # ----------------------------------------------------------------------------

    Agent.alpha = i
    Agent.gamma = j
    Agent.used_in_epsilon = k

    # ----------------------------------------------------------------------------

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    # print

    # print 'contents of Q-table: {}'.format(a.Q)      # Pushpendra, this line is for Debugging purpose
    # print
    # print 'size of Q-table: {}'.format(len(a.Q))     # For OPTIONAL Question  - Pushpendra
    # print

    # print 'True count in self.successes[] : {}'.format(a.successes.count(True))
    # print
    # print 'self.successes[] : {}'.format(a.successes)
    # print
    # print 'Total no. of times when optimal path was found: {}'.format(a.optimal_path_found.count(1))
    # print
    # print 'self.optimal_path_found[]: {}'.format(a.optimal_path_found)
    # print

    return (a.successes, a.optimal_path_found)          # used to fill csv file 'parameter_evaluation.csv'


if __name__ == '__main__':
    list_alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    list_gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    list_epsilon = [1.1, 1.3, 1.5]

    out = open('parameter_evaluation.csv','w')
    column = ('alpha', 'gamma', 'success ratio', 'optimal path found ratio', 'epsilon function')
    out.write('%s,%s,%s,%s,%s' %column)
    out.write('\n')
    out.close()
    out = open('parameter_evaluation.csv','a')

    for i in list_alpha:
        for j in list_gamma:
            for k in list_epsilon:
                success_list, optimal_path_found_list = run(i, j, k)                               # run(arg1, arg2)
                success_ratio = success_list.count(True) / (1.0 * len(success_list))
                ratio_of_number_of_times_optimal_path_found = optimal_path_found_list.count(1) / (1.0 * len(optimal_path_found_list))
                var_epsilon = '1/math.pow((self.total_num_of_trials)(1/'+'{}'.format(k)+'))'
                row = (i, j, success_ratio, ratio_of_number_of_times_optimal_path_found, var_epsilon)
                out.write('%f,%f,%f,%f,%s' %row)
                out.write('\n')

