
class TowerAssembly:
    def __init__(self):
        self.STO = 0
        self.BIN = 1
        self.TAB = 2

        self.num_blocks = 7
        #block colors: 0=red,1=yellow,2=green,3=blue,4=purple,5=grey,6=black
        self.initial_state = ((1,0,0),(1,0,0),(1,0,0),(1,0,0),(1,0,0),\
                (1,0,0),(1,0,0))
        self.goal_states = [ \
                           ((0,0,7),(0,0,3),(0,0,2),(0,0,4),(0,0,5),\
                            (0,0,1),(0,0,6)), \
                           ((0,0,7),(0,0,4),(0,0,2),(0,0,5),(0,0,1),\
                            (0,0,3),(0,0,6)), \
                           ((0,0,4),(0,0,6),(0,0,1),(0,0,7),(0,0,3),\
                            (0,0,2),(0,0,5))]

        #block actions: 0=pickup_storage,1=place_table,2=idle,3=remove_table,
        #4=remove_bin
        self.num_block_actions = 5
        self.a_dict = {}
        for i in range(self.num_blocks):
            for j in range(self.num_block_actions):
                self.a_dict[(i * self.num_block_actions) + j] = (i, j)

        self.terminal_state = -1

    def get_table_height(self, state):
        h = 0
        for s_b in state:
            h = max(h, s_b[self.TAB])
        return h

    def get_num_actions(self):
        return len(self.a_dict.keys())

    def get_top_block(self, state):
        max_h = 0
        top_b = -1
        for i, s_b in enumerate(state):
            if max_h < s_b[self.TAB]:
                max_h = s_b[self.TAB]
                top_b = i
        return top_b

    def get_table_stack(self, state):
        stack = [-1, -1, -1, -1]
        for i, s_b in enumerate(state):
            if s_b[self.TAB] > 0:
                stack[s_b[self.TAB] - 1] = i
        return stack

    def set_block_state(self, state, block, block_state):
        l = list(state)
        l[block] = block_state
        return tuple(l)

    def internal_act(self, block, action, s, g=None):
        if g == None and (s in self.goal_states or s == self.terminal_state):
            return self.terminal_state
        elif (s == g or s == self.terminal_state):
            return self.terminal_state

        h = self.get_table_height(s)

        #block actions: 0=pickup_storage,1=place_table,2=idle,3=remove_table,
        #4=remove_bin
        #if action is pickup from storage put in bin
        if action == 0 and s[block][self.STO] == 1:
            return self.set_block_state(s, block, (0, 1, 0))
        #if action is place on table
        elif action == 1 and s[block][self.BIN] == 1 and h < self.num_blocks:
            return self.set_block_state(s, block, (0, 0, h + 1))
        #if action is idle
        elif action == 2:
            return s
        #if action is remove from table
        elif action == 3 and self.get_top_block(s) == block:
            return self.set_block_state(s, block, (0, 1, 0))
        #if action is remove from bin
        elif action == 4 and s[block][self.BIN] == 1:
            return self.set_block_state(s, block, (1, 0, 0))

        #none of the conditions are met (invalid move from state or in general)
        return s

    def get_all_possible_states(self):
        #all_combinations = self.get_all_possible_state_combinations()
        visited = set([self.initial_state])
        explore = [self.initial_state]
        while len(explore) > 0:
            s_curr = explore.pop()
            for a in self.a_dict.keys():
                block, action = self.a_dict[a]
                s_new = self.internal_act(block, action, s_curr)
                if s_new not in visited and s_new != self.terminal_state:
                    visited.add(s_new)
                    explore.append(s_new)

        return visited

    def get_state_enumeration(self):
        states = self.get_all_possible_states()
        count = 0
        d = {}
        r = {}
        for s in states:
            d[count] = s
            r[s] = count
            count += 1

        self.num_to_state = d
        self.state_to_num = r
        self.terminal_state = len(self.num_to_state.keys())
        self.num_states = self.terminal_state + 1
        return d

    def act(self, a_num, s_num, g_num=None):
        if s_num == self.terminal_state:
            return self.terminal_state
        try:
            s = self.num_to_state[s_num]
        except:
            self.get_state_enumeration()
            s = self.num_to_state[s_num]
        block, action = self.a_dict[a_num]
        s_new = None
        if g_num != None:
            s_new = self.internal_act(block, action, s, g=self.num_to_state[g_num])
        else:
            s_new = self.internal_act(block, action, s, g=None)
        if s_new == self.terminal_state:
            return self.terminal_state
        else:
            return self.state_to_num[s_new]

    def state_rewards(self, s_num, g_num):
        if s_num == g_num:
            return 100
        elif s_num == self.terminal_state:
            return 0
        s = self.num_to_state[s_num]
        g = self.num_to_state[g_num]
        penalty = -8
        tower = [-1, -1, -1, -1, -1, -1, -1]
        for i, b_s in enumerate(s):
            if b_s[self.TAB] != 0:
                tower[b_s[self.TAB] - 1] = 1

        i = 0
        while i < len(tower) and tower[i] != -1:
            penalty += 1
            i += 1

        return penalty
