# UoM COMP90054 Contest Project

**TOC**

- [Youtube presentation](## Youtube presentation)
- [Team Members](## Team Members)
- [Agent Planning ](## Agent Planning)
  * [ Design Decisions](###Design Decisions)
  * [Agent Performance](### Agent Performance)
- [AI Techniques](## AI Techniques )
  * [Heuristic Algorithms](### Heuristic Algorithms)
  * [MDPs](### MDPs)
  * [Challenges experienced about PDDL](### Challenges experienced about PDDL)
- [Possible improvements](## Possible improvements)



## Youtube presentation


![Video link](https://youtu.be/shG-aZ1ShL4)

<figure class="video_container">

  <iframe width="560" height="440" src="https://www.youtube.com/embed/shG-aZ1ShL4" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

</figure>


## Team Members

* Zicong Li - [z.li125@student.unimelb.edu.au](/Users/mac/Library/Application Support/typora-user-images/6C0B9377-FECD-48E1-9026-8FD5F980ACBF/mailto:z.li125@student.unimelb.edu.au) - 807298
* Tao Jin - [tjin2@student.unimelb.edu.au](/Users/mac/Library/Application Support/typora-user-images/CBC39C1C-079C-4837-94C1-6F714E613062/mailto:tjin2@student.unimelb.edu.au) -827872 
* Fanqi He - [f.he6@student.unimelb.edu.au](/Users/mac/Library/Application Support/typora-user-images/251958A3-BE69-4BF9-82C2-A81800B02613/mailto:f.he6@student.unimelb.edu.au) - 880099 

## Agent Planning 

### Design Decisions

We creates two  Pacman agents with different responsibility. At the start of  the game, two agent will move form initial position to the centre of the map, when they arrive at the centre, the attack agent will move to the opposites map to eat food and avoid enemis, the defend agent will stay the centre line to defense and hunt the enemis.

**Start**: We use the function getMazeDistance to get the shortest path from satrt to the center line, because if agents reach the centre earlier than the other team, our agents can begin to attack quickly. 

### Agent Performance

|                     | Attack Agent                                                 | **Defend Agent**                                             |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Food Strategy**   | Attack by moving towards the closest food, the attck agent will judge the current state, if it a safe state then the agent will continute to eat food. | Do not go over the centerline, so do not eat opposite food, the main task is to protect our own team‘s food |
| **Enemy Strategy**  | If the attack agent notices an enemy, it begins moving away from it and still searching food, and if it meets a enemy, the agent will start avoiding the enemy directly and go back to its own side for safety and to deposit food. | Once an enemy eats the food on our side, the defend agent starts to hunt the enemy, and if there are two enemies reach our side, the defend agent will hunt for the closest one. |
| **Defend Strategy** | When the agent is avoiding the enemy, it will search for the power capsule nearby, the agent will eat the power capsule for defence. | If the enemy eats a power capsule, the agent will stay one space away for safety. Once the powers is over, the defend agent can kill the enemy immediately. |

We chose a conservative strategy for agent planning, that is to assign two agents with different tasks, because in this game  we not only considers how to eat more  enemy’s food, but also to protect our own field's food. However the offensive and defensive ability is not the strongest due to the decentralization of responsibility, once the we meet a strong opponents, this strategy may be too conservative to get more points.

## AI Techniques

### Heuristic Algorithms

We choose **Hill Clambing** to return a state that is a local maximum.

* Define different heuristic weightings for different features, here is a example for attack agent

   ```python
   def attack_weight(self, gameState, action):
           return {'successor': 800, 'foodDist': -10, 'enemyClose': -1000,
                   'pickupFood': 4000, 'capsuleDist': 700, 'stop': -1000, 'dead': -50,
                   'isPowered': 999999, 'dropFood': 600, 'holdFood': -20,
                   'distToAlly': -6000, 'pickupCapsule': 5000}
   ```

* For current state, evaluate all the possible actions to calculate the heurisic score, and then pick a optimal heuristic

   **Best Action:**

   - Get all legal actions this agent can make in current state
   - Calcualte heuristic score of each action
   - From the current state, compare the heuristic score 
   - Pick the action with the highest heuristic score as the best next move

   ```python
   actions = gameState.getLegalActions(self.index)
           # Calculate the heuristic score
           values = []
           for action in actions:
               values.append(self.evaluate(gameState, action, type_eval))
           maxValue = max(values)
           bestAction = []
           for action, value in zip(actions, values):
               if value == maxValue:
                   bestAction.append(action)
   
           return bestAction[0]
   ```

### MDPs

Based on the lecture, V(s) is the expected value of being in state s and acting optimally according to our policy, then we can also describe the Q-value of being in a state s, choosing action a and then acting optimally according to our policy as Q(a,s).


We ues a global variable beliefs to store the possibility of enemies.

```python
# get probabilities of enemies
        global beliefs
        beliefs = [util.Counter()] * gameState.getNumAgents()
        for i, val in enumerate(beliefs):
            if i in self.enemies:
                beliefs[i][gameState.getInitialAgentPosition(i)] = 1
```

### Challenges experienced about PDDL

PDDL, the Planning Domain Definition Language, describes the initial and goal states as conjunctions of literals, and actions in terms of their preconditions and effects. At first we try to use this Classical Planning method to determine each action, here is an example of using PDDL:

```python
 action = 'Stop'
    data = {'domain': open("pacman-domain.pddl", 'r').read(),
        'problem': open("problem" + str(self.index + 1) + ".pddl", 'r').read()}
    
    req = urllib2.Request('http://solver.planning.domains/solve')
    req.add_header('Content-Type', 'application/json')
    resp = json.loads(urllib2.urlopen(req, json.dumps(data)).read())
    
    for act in resp['result']['plan']:
        s = act['name']
        #print s
        pos = s.find("MOVE")
        #print pos
        if pos < 0:
       ···
```

This method uses online solver, the frequent I/O operations of read and write files cause low performance of the whole project. So we decide to give up PDDL.

## Possible improvements

* For PDDL, the possible improvements is that we can use the third party solver or local solver to optimize PDDL algorithm's performance.
* In  the current project, the feature weight are defined manual based on knowledge experience,  the possible improbement is that we can use Q learing or other Reinforcement Learning techniques to determine the weight of each feature.









