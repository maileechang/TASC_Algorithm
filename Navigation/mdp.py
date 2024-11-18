#!/usr/bin/env python

import mdptoolbox
import numpy as np
import math
from copy import deepcopy

"""
    This class implements an MDP for a two-goal 10x10 state space
"""
class MDP:
    def __init__(self):
        #define state space
        self.actions = 6 #number of possible actions
        self.l = 10 #the state space is 10x10
        self.states = (self.l**2) + 1 #total number of states
        self.goals = [92,98] #goal states
        self.obstacles = [64] #where obstacles are located
        
        #initialize transition matrix
        self.P = np.zeros([self.actions,self.states,self.states])
        
        #initialize reward for both the human and the robot
        self.R_human = -1 * np.ones(self.states)
        self.R_robot = -1 * np.ones([self.states])
        
        #set gamma for learning
        self.gamma = 0.9
        
        #initialize values states
        self.V = None
        self.Vs_robot = {}
        self.Vs_human = {}
        
        #initialize policy storage
        self.policy = None
        self.policies = {}
        
        #set up mdp information
        self.setup()
        #print("state space", self.states)
        
        
       
    """
        This function makes the rewards for the human teammate
    """
    def make_rewards_human(self):
        #reward 100 for reaching any goal
        for g in self.goals:
            self.R_human[g] = 100
     
    """
        This function makes the rewards for the robot teammate
    """
    def make_rewards(self):
        #reward 100 for reaching any goal
        for g in self.goals:
            self.R_robot[g] = 100
        
    """
        This function returns the next state given the current state and action
    """     
    def act(self,a,s):
        #if s is a goal state or the terminal state, go to terminal state no matter what action is taken
        if s in self.goals or s == self.states - 1:
            return self.states - 1
        #if a=0, try to move north
        elif a == 0 and s + self.l < self.states-1 and s + self.l not in self.obstacles:
            return s + self.l
        #if a=1, try to move northeast
        elif a == 1 and s + self.l + 1 < self.states-1 and (s%self.l) < (self.l-1) and s + self.l + 1 not in self.obstacles:
            return s + self.l + 1
        #if a=2, try to move east
        elif a == 2 and s + 1 < self.states-1 and (s%self.l) < (self.l-1) and s + 1 not in self.obstacles:
            return s + 1
        #if a=3, try to move west
        elif a == 3 and s-1 > 0 and (s%self.l) > 0 and s - 1 not in self.obstacles:
            return s - 1
        #if a=3, try to move northwest
        elif a == 4 and s + self.l - 1 < self.states-1 and (s%self.l) > 0 and s + self.l - 1 not in self.obstacles:
            return s + self.l - 1
        #if a=5, stay
        elif a == 5:
            return s
        #catchall
        return s
       
    """
        This function makes the transition probability matrix
    """ 
    def make_transition(self):
        for a in range(self.actions):
            for s in range(self.states):
                #no probability of failure, so 100% chance of transitioning to desired state
                self.P[a,s,self.act(a,s)] = 1
            
    """
        This function sets up the rewards and the transition function
    """   
    def setup(self):
        self.make_rewards_human()
        self.make_transition()
        self.value_iter_human()
        
        self.make_rewards()
        self.value_iter()


    def square(self,s):
        return(s%self.l,int(s/self.l))                   
        
    """
        This function does value iteration on human's mdp
    """
    def value_iter_human(self):
        vis = []
        #for all possible goals
        for g in self.goals:
            #save a copy of the human's rewards
            R_copy_human = deepcopy(self.R_human)
            #for all goals besides g, set the reward to -1
            for g_other in self.goals:
                if g != g_other:
                    R_copy_human[g_other] = -1
            #solve values for when the human's goal is g
            vi_copy_human = mdptoolbox.mdp.ValueIteration(self.P, R_copy_human, self.gamma)
            vi_copy_human.run()
            
            #save values and policies in dictionary indexed by goal
            self.Vs_human[g] = vi_copy_human.V
            self.policies[g] = vi_copy_human.policy
            print("human values:", self.Vs_human[g])
            print("human policies:", self.policies[g])
            #print("human values:", len(self.Vs_human[92]))
            #print("human policies:", len(self.policies[92]))
        for s in range(100):
            print(str(self.square(s)) + " " + str(self.policies[92][s]))
    
    """
        This function does value iteration on the robot's mdp
    """
    def value_iter(self):
        #solve values and policies for the robot for all goals set to 100 reward
        vi = mdptoolbox.mdp.ValueIteration(self.P, self.R_robot, self.gamma)
        vi.run()
        self.V = vi.V
        self.policy = vi.policy
        vis = []
        #for all possible goals
        for g in self.goals:
            #copy the robot's rewards
            R_copy_robot = deepcopy(self.R_robot)
            #for all goals besides g, set the reward to -1
            for g_other in self.goals:
                if g != g_other:
                    R_copy_robot[g_other] = -1
            
            #solve values for when the goal is g
            vi_copy_robot = mdptoolbox.mdp.ValueIteration(self.P, R_copy_robot, self.gamma)
            vi_copy_robot.run()
            
            #save values in dictionary indexed by goal
            self.Vs_robot[g] = vi_copy_robot.V

        for s in range(100):
            print(str(self.square(s)) + " " + str(self.Vs_robot[92][s]))        
    
      
        
def main():
    mdp = MDP()
    
main()
