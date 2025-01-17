# TASC: Teammate Algorithm for Shared Cooperation

This respository contains code for the following paper:

[Chang, M. L., Faulkner, T. K., Wei, T. B., Short, E. S., Anandaraman, G., & Thomaz, A. L. (2020, October). TASC: Teammate Algorithm for Shared Cooperation. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (pp. 11229-11236). IEEE.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9340983)

## Navigation and Modified Navigation Tasks
### 1. Navigate to the Navigation directory and in ```mdp.py```, set the goal states (self.goals) and obstacles (self.obstacles)
### 2. In ```TASC_nav.py```, set weights on value, effort, and legibility (wV, wE, wL)
### 3. Run ```TASC_nv.py``` to get policies for robot and human teammate
### 4. For the navigation task, input polices in ```MTurk_Nav.html```. For the modified navigation task, input polices in ```MTurk_Modified_Nav.html```. 
### 5. User Studies: Run ```MTurk_Nav.html```, ```MTurk_Modified_Nav.html``` on Google Chrome

## Tower Assembly Task
### 1. Navigate to the Tower_Assembly directory and in ```TASC_tower.py```, set weights on value, effort, and legibility (wV, wE, wL)
### 2. Run ```TASC_tower.py``` to get policies for robot and human teammate
### 3. Input polices in ```MTurk_towers_effort.html```
### 4. User Study: Run ```MTurk_towers_effort.html``` on Google Chrome

## Note:
Due to IRB restrictions, we are not able to share our data. 

## Citation
If you find this repository is useful in your research, please cite the paper:
```
@inproceedings{chang2020tasc,
  title={TASC: Teammate Algorithm for Shared Cooperation},
  author={Chang, Mai Lee and Faulkner, Taylor Kessler and Wei, Thomas Benjamin and Short, Elaine Schaertl and Anandaraman, Gokul and Thomaz, Andrea Lockerd},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={11229--11236},
  year={2020},
  organization={IEEE}
}
```
