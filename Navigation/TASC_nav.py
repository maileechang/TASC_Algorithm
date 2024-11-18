#!/usr/bin/env python

"""
Run this program to use the TASC or SCA algorithm on an MDP problem (Navigation).
Prints state-action trajectory through MDP, with predicted
goal probabilities.
"""

import numpy as np
from mdp import MDP
from scipy.spatial.distance import euclidean
import random


"""
    This class implements the SCA algorithm
"""
class SCA:
    def __init__(self, wV=0.9, wE=0.05, wL=0.05):
        random.seed()

        #create instance of problem MDP
        self.mdp = MDP()

        #pull out variables from MDP
        self.S = self.mdp.states #number of states
        self.AH = self.mdp.actions #number of human actions
        self.AR = self.mdp.actions #number of robot actions
        self.G = self.mdp.goals #index of possible goal states


        self.Gp = None #predicted goal
        self.V = self.mdp.V #state values from MDP
        self.Vs_human = self.mdp.Vs_human #dictionary indexed by goal state. Gives state values from perspective of human (planned goal)
        self.Vs_robot = self.mdp.Vs_robot #dictionary indexed by predicted goal state. Gives state values from perspective of robot (predicted goal)
        self.policies = self.mdp.policies #policies learned in MDP

        #find maximum state value
        self.maxV = self.V[0]
        for v in self.V:
            self.maxV = max(v,self.maxV)

        #find maximum state value given robot's predicted goal
        self.maxVs = {}
        for G in self.Vs_robot:
            for v in self.Vs_robot[G]:
                self.maxVs[G] = max(v,self.maxV)

        #set weights for Value, Effort, and Legibility
        self.wV = wV
        self.wE = wE
        self.wL = wL

        #set current state and previous state
        self.s = None
        self.s_old = None

        #set human action and goal
        self.aH = None
        self.human_goal = self.G[0]

        #for printing
        self.move_strings = {0 : 'up', 1 : 'up right', 2 : 'right', 3 : 'left', 4 : 'up left', 5 : 'idle'}

    """
        This function returns the index location of a state in the mdp state space
    """
    def square(self,s):
        return(s%self.mdp.l,int(s/self.mdp.l))


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
        This function returns the legibility probability of goal G given a robot action a
    """
    def PrG(self,G,a):
        #set g to the goal that is not G (assumes two goals)
        if G == self.G[0]:
            g = self.G[1]
        else:
            g = self.G[0]

        #s is current state
        s = self.s
        #s_new is predicted new state given action a
        s_new = self.mdp.act(a, self.s)

        #d is the euclidean distance between the indices of G and s minus
        #the euclidean distance between the indices of G and s_new
        d = euclidean(self.square(G),self.square(s)) - euclidean(self.square(G),self.square(s_new))
        #d2 is the euclidean distance between the indices of g and s minus
        #the euclidean distance between the indices of g and s_new
        d2 = euclidean(self.square(g),self.square(s)) - euclidean(self.square(g),self.square(s_new))

        #if the distances are the same, both goals are equally likely
        if d == d2:
            return 0.5
        #if s_new takes the robot farther from G and closer to g, 0% chance G is the goal
        elif d<=0 and d2>0:
            return 0
        #if s_new takes the robot farther from g and closer to G, 100% chance G is the goal
        elif d > 0 and d2 <= 0:
            return 1
        #otherwise, return a probability based on the distance between the two (could be changed)
        return((d/d2)/((d/d2)+(d2/d)))


    """
        This function predicts the goal state and probability
    """
    def CG(self,a):
        #if the person doesn't take an action, pick a random goal and assign 50% probability
        if a == None:
            r = 1#random.randint(0,len(self.G)-1)
            return (self.G[r],0.5)

        max_g = []
        max_pr = -1

        for g in self.G:
            p = self.PrG(g, a)
            if p > max_pr:
                max_g = [g]
                max_pr = p
            elif p == max_pr:
                max_g.append(g)

        return (max_g[random.randint(0,len(max_g)-1)], max_pr)

        """
        #set up all possible goals
        poss_goals = []

        #look through policies given a goal G
        for G in self.policies:
            #if the action to take at state s_old matches the action the human took, add goal G to the possible goals
            if self.policies[G][self.s_old] == a:
                poss_goals.append(G)

        #if there is only one goal, give it a probability of 1
        if len(poss_goals) == 1:
            p = 1.0
            return (poss_goals[0],p)
        #otherwise, mark possibility at 50/50
        else:
            p = 0.5

        #if no predicted goal, or the predicted goal isn't in the possible goals, return a random possible goal
        if self.Gp == None or self.Gp not in poss_goals:
            r = random.randint(0,len(poss_goals)-1)

            return (poss_goals[r],p)
        return (self.Gp,p)
        """

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
        s_new = self.mdp.act(self.aH, self.s)
        #print("s_new:", s_new)

        #append the indices of the new visited states to the solution (first human action, then robot action)
        sol.append((self.square(self.mdp.act(self.aH,self.s)), 'H'))
        print("STATE human:",self.square(self.s)," AH:", self.move_strings[self.aH])

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
        (self.Gp,p) = self.CG(self.aH)
        ap = self.CA(self.Gp)
        print("Predicted goal:",self.Gp, " Prob:", p)

        #look through all possible actions
        for a in range(self.AR):
            #see what the next state would be
            s_new = self.mdp.act(a, self.s)
            #see what the next state would be after one predicted action by human
            #s_one = self.mdp.act(ap,self.s)
            #calculate probability of effort
            E = self.PrE(a, self.s, s_new)

            """
            #if the probability of both goals is 0, then legibility is 0
            if ((self.PrG(self.G[0],a) + self.PrG(self.G[1],a))) == 0:
                L = 0
            #otherwise if both goals are possible
            elif p == 0.5:
                #calculate legibility
                L = (0.5*self.PrG(self.G[0],a) + 0.5*self.PrG(self.G[1],a) + 1 - abs(self.PrG(self.G[0],a)-self.PrG(self.G[1],a)))/2
            #otherwise if only one goal is possible
            else:
                #calculate legibility
                L = (self.PrG(self.G[0],a) + 1 - abs(self.PrG(self.G[0],a)-self.PrG(self.G[1],a)- 1))/2
            """
            if self.Gp == 92:
                L = p * self.PrG(self.Gp, a) + (1 - p) * self.PrG(98, a)
            else:
                L = p * self.PrG(self.Gp, a) + (1 - p) * self.PrG(92, a)
            print(str(self.move_strings[a]) + " " + str(L))

            #calculate value of new state for both goals
            v0 = (self.Vs_robot[self.G[0]][s_new]/self.maxVs[self.G[0]])
            v1 = (self.Vs_robot[self.G[1]][s_new]/self.maxVs[self.G[1]])
            """
            #if value of both is zero, set V to 0
            if (v0 + v1 == 0):
                V = 0
            #else if both goals are possible, calculate V
            elif p == 0.5:
                V = (0.5*v0+0.5*v1+1 - abs(v0-v1))/2
            #else if only one goal is possible, calculate V
            else:
                V = (v0+1 - abs(v0-v1- 1))/2
            """
            if self.Gp == 92:
                V = p * v0 + (1 - p) * v1
            else:
                V = p * v1 + (1 - p) * v0

            #combined value of Effort, Legibility, and Value
            val = self.wE*E + self.wL*L + self.wV*V

            #if the value of this action is greater than previously seen, save it
            if val > mx:
                mx = val
                maxes = [a]
            elif val == mx:
                maxes.append(a)

        #choose a random one of the max valued actions
        aR = maxes[random.randint(0,len(maxes)-1)]


        sol.append((self.square(self.mdp.act(aR,self.s)), 'R'))
        print("STATE robot:",self.square(self.s)," AR:", self.move_strings[aR])
        s_new = self.mdp.act(aR,self.s)

        #save old state, new state
        self.s = s_new


    """
        This function picks the actions for each teammate
    """
    def team(self, h_act, r_act):
        #start at a random state (or choose a state here)
        self.s = random.randint(0,self.mdp.l-1)
        #for these experiments I started at state 4
        self.s = 4

        #start at time 0
        t = 0

        #sol is the final decided state trajectory
        sol = []
        #start by appending the indices of the initial state
        sol.append(self.square(self.s))
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



"""
    Run SCA algorithm
"""
def main():
    #create instance of SCA, run the team algorithm
    sca = SCA()
    h_act = lambda x: True if x % 2 != 0 else False
    r_act = lambda x: True if x % 2 == 0 else False
    sca.team(h_act, r_act)


main()
