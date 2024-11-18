#!/usr/bin/env python

"""
Run this program to use the TASC or SCA algorithm on an MDP problem (tower assembly).
Prints state-action trajectory through MDP, with predicted
goal probabilities.
"""

import numpy as np
import random
import pickle

from io import BytesIO
from scipy.spatial.distance import euclidean
from tower_assembly import TowerAssembly

"""
    This class implements the SCA algorithm
"""
class SCA:
    def __init__(self, wV=0.9, wE=0.05, wL=0.05, human_goal=0):
        np.random.seed(1)

        #create instance of tower assembly and load MDP data
        self.t = TowerAssembly()
        self.t.num_to_state = pickle.load(open('num_to_state.pkl', 'rb'))
        self.t.state_to_num = pickle.load(open('state_to_num.pkl', 'rb'))
        self.t.terminal_state = len(self.t.num_to_state.keys())
        self.S = len(self.t.num_to_state.keys()) + 1 #number of states
        self.G = pickle.load(open('goals.pkl', 'rb'))
        self.AH = self.t.get_num_actions() #number of human actions
        self.AR = self.t.get_num_actions() #number of robot actions

        self.Gp = None #predicted goal
        self.Vs_human = pickle.load(open('Vs_human.pkl', 'rb')) #dictionary indexed by goal state. Gives state values from perspective of human (planned goal)
        #self.Vs_robot = pickle.load(open('Vs_robot.pkl', 'rb')) #dictionary indexed by predicted goal state. Gives state values from perspective of robot (predicted goal)
        self.Vs_robot = pickle.load(open('Vs_human.pkl', 'rb')) #dictionary indexed by predicted goal state. Gives state values from perspective of robot (predicted goal)
        self.policies = pickle.load(open('policies.pkl', 'rb')) #policies learned in MDP

        #find maximum state value given robot's predicted goal
        self.maxVs = {}
        for G in self.Vs_robot:
            self.maxVs[G] = 0
            for v in self.Vs_robot[G]:
                self.maxVs[G] = max(v,self.maxVs[G])

        #set weights for Value, Effort, and Legibility
        self.wV = wV
        self.wE = wE
        self.wL = wL

        #set current state and previous state
        self.s = None
        self.s_old = None

        #set human action and goal
        self.aH = None
        self.human_goal = self.G[human_goal] #CHANGE

        #for printing
        self.move_strings = {}
        for i in range(self.t.get_num_actions()):
            self.move_strings[i] = str(self.t.a_dict[i])

    """
        Thus function is for writing the solution in terms of state rather than action
    """
    def num_to_output(self, s):
        if s == self.t.terminal_state:
            return self.t.terminal_state
        return self.t.num_to_state[s]

    """
        This function returns the probability of perceived effort given a previous
        state, action, and current state
    """
    def PrE(self,a,s,s_new):
        #if no movement, probability is low
        if s == s_new:
            return 0.1
        #if movement, probability is high
        else:
            return 0.9

    """
        Simple distance function between two states in this tower assembly problem
    """
    def dist(self, g, s):
        d = 0
        for i in range(self.t.num_blocks):
            if g[i][0] == 1:
                d += abs(g[i][0] - s[i][0])
            elif g[i][1] == 1:
                d += abs(g[i][1] - s[i][1])
            elif g[i][2] != 0 and (s[i][0] == 1 or (s[i][2] != 0 and s[i][2] != g[i][2])):
                d += 2
            elif g[i][2] != 0 and s[i][1] == 1:
                d += 1
        return d

    """
        This function returns the legibility probability of goal G given a robot action a
    """
    def PrG(self,G,a,s_num=None):
        if s_num == None:
            s_num = self.s
        s = self.t.num_to_state[s_num]
        g = self.t.num_to_state[G]

        #s_new is predicted new state given action a
        s_new = self.t.num_to_state[self.t.act(a, s_num, g_num=self.human_goal)]

        #rudimentary distance metric between states
        d_G = self.dist(g, s) - self.dist(g, s_new)

        #if the move is away from the goal, return probability of 0
        if d_G < 0:
            return 0

        #calculate euclidean distance difference measure for all goals
        #keep track of total sum of ds for normalization purposes
        sum_dist = 0
        for i in self.G:
            g = self.t.num_to_state[i]
            d = self.dist(g, s) - self.dist(g, s_new)

            #disregard negative and 0 ds
            if d > 0:
                sum_dist +=d

        #if nothing changed (idle)
        if sum_dist == 0:
            return 0

        return (d_G/sum_dist)

    """
        This function predicts the goal state and probability based off of
        the differences in the values of the states in the solved MDP
        for each goal (maybe normalize each mdp state value?)
    """
    def CG_markov(self, a):
        eq_p = 1.0 / len(self.G)
        eq_probs = [eq_p for i in range(len(self.G))]

        #if the person doesn't take an action, pick a random goal and assign equal % probability
        if a == None:
            print(eq_probs)
            r = random.randint(0,len(self.G)-1)
            return (self.G[r], eq_p, eq_probs)

        max_g = []
        max_val = 0

        #keep running sum of values for normalization
        action_values = []
        sum_vals = 0

        non_neg = []
        #iterate through goals, calculating the difference in mdp state value
        #caused by each action
        for g in self.G:
            val = self.Vs_human[g][self.s] - self.Vs_human[g][self.s_old]

            #maintain maximum difference goal mdp
            if val > max_val:
                max_g = [g]
                max_val = val
            elif val == max_val:
                max_g.append(g)

            #consider only positive differences
            if val >= 0:
                sum_vals += val
                non_neg.append(True)
            else:
                non_neg.append(False)

            action_values.append(val)

        if sum_vals == 0:
            max_g = []
            num = 0
            for b in non_neg:
                if b:
                    num += 1
            if num > 0:
                eq_p = 1 / num
                for i in range(len(non_neg)):
                    if non_neg[i]:
                        eq_probs[i] = eq_p
                        max_g.append(self.G[i])
                    else:
                        eq_probs[i] = 0
            return (max_g[random.randint(0,len(max_g)-1)], eq_p, eq_probs)

        #normalize the (positive) difference values for each goal mdp
        #assign probability 0 if the difference is negative
        probs = [val / sum_vals if val > 0 else 0 for val in action_values]
        max_pr = max(max_val / sum_vals, eq_p)

        return (max_g[random.randint(0,len(max_g)-1)], max_pr, probs)


    """
        This function returns an action prediction based on the goal
    """
    def CA(self,G):
        #gather possible actions based on the learned policies given each possible goal
        poss_actions = []
        poss_actions.append(self.policies[G][self.s])

        #return a random possible action
        r = random.randint(0,len(poss_actions)-1)

        return poss_actions[r]


    def human_action(self, sol):
        #choose human action based on the policies
        self.aH = self.policies[self.human_goal][self.s]
        #calculate new state
        s_new = self.t.act(self.aH, self.s, g_num=self.human_goal)
        #print("s_new:", s_new)

        #append the indices of the new visited states to the solution (first human action, then robot action)
        sol.append((self.num_to_output(self.t.act(self.aH,self.s, g_num=self.human_goal)), 'H'))
        print("STATE human:",self.num_to_output(self.s)," AH:", self.move_strings[self.aH])

        #save old state, new state
        self.s_old = self.s
        self.s = s_new


    def robot_action(self, sol):
        #collect maximum value options for robot actions
        mx = -np.inf
        maxes = []

        #robot action
        aR = None

        #predict the human's goal and action
        #self.Gp, p, probs = self.CG_euclid(self.aH)
        self.Gp, p, probs = self.CG_markov(self.aH)
        ap = self.CA(self.Gp)
        print("Predicted goal:", self.t.num_to_state[self.Gp] , " Prob:", p)
        print("Probs: " + str(probs))

        V_list = []
        for a in range(self.AR):
            s_new = self.t.act(a, self.s, g_num=self.human_goal)
            for i, g in enumerate(self.G):
                V_g = self.Vs_robot[g][s_new] - self.Vs_robot[g][self.s]
                V_list.append(V_g)
        max_V = np.max(np.absolute(V_list))

        #look through all possible actions
        for a in range(self.AR):
            #see what the next state would be
            s_new = self.t.act(a, self.s, g_num=self.human_goal)
            #see what the next state would be after one predicted action by human
            #s_one = self.mdp.act(ap,self.s)
            #calculate probability of effort
            E = self.PrE(a, self.s, s_new)

            #calculate probability that action a will be percieved as towards
            #predicted goal
            L = self.PrG(self.Gp, a)

            #calculate expected value of new state
            V = 0
            if max_V > 0:
                for i, g in enumerate(self.G):
                    V += probs[i] * ((self.Vs_robot[g][s_new] - self.Vs_robot[g][self.s]) / max_V)
                #normalize
                V = (V/2) + 0.5

            #combined value of Effort, Legibility, and Value
            val = self.wE*E + self.wL*L + self.wV*V
#            print('Probs ' + str(probs))
            print('Val ' + str(self.t.a_dict[a]) + " " + str(val))

            #if the value of this action is greater than previously seen, save it
            if val > mx:
                mx = val
                maxes = [a]
            elif val == mx:
                maxes.append(a)

        #choose a random one of the max valued actions
        aR = maxes[random.randint(0,len(maxes)-1)]


        sol.append((self.num_to_output(self.t.act(aR,self.s, g_num=self.human_goal)), 'R'))
        print("STATE robot:",self.num_to_output(self.s)," AR:", self.move_strings[aR])
        s_new = self.t.act(aR,self.s, g_num=self.human_goal)

        #save old state, new state
        self.s = s_new


    """
        This function picks the actions for each teammate
    """
    def team(self, h_act, r_act, s=None):
        if s == None:
            self.s = self.t.state_to_num[self.t.initial_state]
        self.s_old = None
        #start at time 0
        t = 0

        #sol is the final decided state trajectory
        sol = []

        #start by appending the indices of the initial state
        sol.append(self.num_to_output(self.s))
        #while not in the last state (a terminal state that all goals lead to)
        while self.s != self.S - 1:
            if r_act(t) == True:
                self.robot_action(sol)
            if h_act(t) == True:
                self.human_action(sol)

            #increase time by 1
            t += 1
            print(t)

        #print the total solution
        print(sol)

        return sol

    def game_init(self, s=None):
        if s == None:
            self.s = self.t.state_to_num[self.t.initial_state]
        self.s_old = None
        #start at time 0
        t = 0

        #sol is the final decided state trajectory
        self.game_sol = []
        #start by appending the indices of the initial state
        self.game_sol.append(self.num_to_output(self.s))
        return self.game_sol

    def game_robot_step(self):
        #while not in the last state (a terminal state that all goals lead to)
        if self.s != self.S - 1:
                self.robot_action(self.game_sol)
        return self.game_sol

    def game_human_step(self, a):
        if self.s != self.S - 1:
                s_new = self.t.act(a,self.s, g_num=self.human_goal)
                self.game_sol.append((self.num_to_output(s_new), 'H'))
                print("STATE human:",self.num_to_output(self.s)," AH:", self.move_strings[a])

                #save old state, new state
                self.s_old = self.s
                self.s = s_new
                self.aH = a

        return self.game_sol

    def game_auto_step(self):
        #choose action based on the policies
        self.aH = self.policies[self.human_goal][self.s]
        #calculate new state
        s_new = self.t.act(self.aH, self.s, g_num=self.human_goal)

        self.game_sol.append((self.num_to_output(s_new), 'H'))
        print("STATE human:",self.num_to_output(self.s)," AH:", self.move_strings[self.aH])

        #save old state, new state
        self.s_old = self.s
        self.s = s_new
        return self.game_sol

