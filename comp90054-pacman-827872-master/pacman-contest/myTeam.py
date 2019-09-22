# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

#AI project2
# 827872 TaoJin
# 807298 Zicong Li
# 880099 Fanqi He
from captureAgents import CaptureAgent
import random, time, util, operator
from util import nearestPoint
from game import Directions
import game


# Create and initialize a team
def createTeam(firstIndex, secondIndex, isRed,
               first='Attack', second='Defend'):
    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


# A base class for reflex agents that chooses score-maximizing actions
class AttackAgent(CaptureAgent):

    # Initialize the beginning position and power timer for the created agent
    def __init__(self, gameState):
        CaptureAgent.__init__(self, gameState)
        self.pos = [None] * 4
        self.powerTimer = 0

    def registerInitialState(self, gameState):

        CaptureAgent.registerInitialState(self, gameState)

        # Judge team color
        if self.red:
            CaptureAgent.registerTeam(self, gameState.getRedTeamIndices())
        else:
            CaptureAgent.registerTeam(self, gameState.getBlueTeamIndices())

        # Get the board space
        self.x = gameState.data.layout.width
        self.y = gameState.data.layout.height

        # Get possible positions when there are no walls
        self.legalPositions = []
        for p in gameState.getWalls().asList(False):
            if p[1] > 1:
                self.legalPositions.append(p)
        self.walls = list(gameState.getWalls())

        self.chokes = []

        if self.red:
            coolDistance = -3
        else:
            coolDistance = 4
        for i in range(self.y):
            if not self.walls[self.x / 2 + coolDistance][i]:
                self.chokes.append(((self.x / 2 + coolDistance), i))
        # Index of each Agent
        if self.index == max(gameState.getRedTeamIndices()) or self.index == max(gameState.getBlueTeamIndices()):
            x, y = self.chokes[3 * len(self.chokes) / 4]
        else:
            x, y = self.chokes[1 * len(self.chokes) / 4]
        self.goalSetting = (x, y)

        # Get enemies/opponents
        self.enemies = self.getOpponents(gameState)

        # get probabilities of enemies
        global beliefs
        beliefs = [util.Counter()] * gameState.getNumAgents()
        for i, val in enumerate(beliefs):
            if i in self.enemies:
                beliefs[i][gameState.getInitialAgentPosition(i)] = 1

        self.runAgent(gameState)

    # Get position of enemies
    def getEnemies(self, gameState):
        enemyPositions = []
        for enemy in self.enemies:
            position = gameState.getAgentPosition(enemy)
            if position != None:
                enemyPositions.append((enemy, position))
        return enemyPositions

    # Find the closest enemy
    def enemy_distance(self, gameState):
        enemyPositions = self.getEnemies(gameState)
        min_dist = []
        if len(enemyPositions) > 0:
            agentPosition = self.getAgentPosition(gameState)
            for enemy, position in enemyPositions:
                min_dist.append(self.getMazeDistance(position, agentPosition))
                min_dist.sort()
            return min_dist[0]
        else:
            return None

    # Judge whether the state of an agent is the pacman or the ghost
    def isAgentPacman(self, gameState):
        return gameState.getAgentState(self.index).isPacman

    # Get the position of the agent
    def getAgentPosition(self, gameState):
        return gameState.getAgentPosition(self.index)

    # Get the distance between agents
    def get_dist_to_partner(self, gameState):
        distance_to_agent = None
        agents_list = self.agentsOnTeam
        if self.index == agents_list[0]:
            otherAgentIndex = agents_list[1]
            distance_to_agent = None
        else:
            otherAgentIndex = agents_list[0]
            agentPosition = self.getAgentPosition(gameState)
            anotherPosition = gameState.getAgentState(otherAgentIndex).getPosition()
            distance_to_agent = self.getMazeDistance(agentPosition, anotherPosition)
            if distance_to_agent == 0:
                distance_to_agent = 0.1
        return distance_to_agent

    # Judge whether the agent is red or blue
    def side(self, gameState):
        width = gameState.data.layout.width
        pos = gameState.getAgentPosition(self.index)
        if self.index % 2 == 1:
            # red
            if pos[0] < width / (2):
                return 1.0
            else:
                return 0.0
        else:
            # blue
            if pos[0] > width / 2 - 1:
                return 1.0
            else:
                return 0.0

    def isPowered(self):
        return self.powerTimer > 0

    def getDist(self, p):
        pos_actions = [(p[0] - 1, p[1]), (p[0] + 1, p[1]), (p[0], p[1] - 1), (p[0], p[1] + 1),
                       (p[0], p[1])]  # go East,West,North,South and Stop
        actions = []
        for action in pos_actions:
            if action in self.legalPositions:
                actions.append(action)
        distance = util.Counter()
        for action in actions:
            distance[action] = 1
        return distance

    def time_elapsed(self, gameState):
        for agent, belief in enumerate(beliefs):
            if agent in self.enemies:
                newBeliefs = util.Counter()
                # Checks to see what we can actually see
                position = gameState.getAgentPosition(agent)
                if position != None:
                    newBeliefs[position] = 1.0
                else:
                    # Look at all current beliefs
                    for prob in belief:
                        if prob in self.legalPositions and belief[prob] > 0:
                            # Check that all these values are legal positions
                            newPosDist = self.getDist(prob)
                            for x, y in newPosDist:  # iterate over these probabilities
                                newBeliefs[x, y] += belief[prob] * newPosDist[x, y]
                                # The new chance is old chance * prob of this location from prob
                    if len(newBeliefs) == 0:
                        old_state = self.getPreviousObservation()
                        if old_state != None and old_state.getAgentPosition(agent) != None:  # ate an enemy
                            newBeliefs[old_state.getInitialAgentPosition(agent)] = 1.0
                        else:
                            for prob in self.legalPositions: newBeliefs[prob] = 1.0
                beliefs[agent] = newBeliefs

    # Start to find the enemies
    def observe(self, agent, noisyDistance, gameState):
        agent_position = gameState.getAgentPosition(self.index)
        all_possible = util.Counter()
        for position in self.legalPositions:
            true_distance = util.manhattanDistance(position, agent_position)
            all_possible[position] += gameState.getDistanceProb(true_distance, noisyDistance)
            beliefs[agent][position] *= all_possible[position]

    # Get the optimal action
    def chooseAction(self, gameState):
        # Picks actions with the highest value.
        enemies = self.enemies
        noisy_distance = gameState.getAgentDistances()
        for enemy in enemies:
            self.observe(enemy, noisy_distance[enemy], gameState)

        self.locations = [self.chokes[len(self.chokes) / 2]] * gameState.getNumAgents()
        for i, belief in enumerate(beliefs):
            max_loction = 0
            is_all_equal = 0
            for val in beliefs[i]:

                if belief[val] == max_loction and max_loction > 0:

                    is_all_equal += 1
                elif belief[val] > max_loction:
                    max_loction = belief[val]
                    self.locations[i] = val

            if is_all_equal > 5:
                self.locations[i] = self.goalSetting

        # Get the most possible location for enemy
        for enemy in enemies:
            beliefs[enemy].normalize()
            self.pos[enemy] = max(beliefs[enemy].iteritems(), key=operator.itemgetter(1))[0]

        # Next Step
        self.time_elapsed(gameState)
        agent_pos = gameState.getAgentPosition(self.index)

        # Default to attack mode
        type_eval = 'attack'

        # Start in the start state before move to center
        if not self.atCenter:
            type_eval = 'start'

        # If at centre, switch to attack
        if agent_pos == self.center and self.atCenter == False:
            self.atCenter = True
            type_eval = 'attack'

        # Get possible actions based on the current state
        actions = gameState.getLegalActions(self.index)
        # Calculate the heuristic score of each action
        values = []
        for action in actions:
            values.append(self.evaluate(gameState, action, type_eval))

        # Pick the action with the highest heuristic score as the best one
        maxValue = max(values)
        bestAction = []
        for action, value in zip(actions, values):
            if value == maxValue:
                bestAction.append(action)

        return bestAction[0]

    # Get a general Successor
    def getSuccessor(self, gameState, action):
        """
    Find the next successor which is a grid position (location tuple).
    """
        successor = gameState.generateSuccessor(self.index, action)
        position = successor.getAgentState(self.index).getPosition()
        if position != nearestPoint(position):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    # Calculate the heurisic score of an action
    def evaluate(self, gameState, action, type_eval):
        # Calculate the weight of features
        if type_eval == 'attack':
            features = self.attack_feature(gameState, action)
            weights = self.attack_weight(gameState, action)
        elif type_eval == 'start':
            features = self.start_feature(gameState, action)
            weights = self.start_weight(gameState, action)

        return features * weights

    # Get the heuristic features for attack
    def attack_feature(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        agent_state = successor.getAgentState(self.index)
        agentPosition = agent_state.getPosition()
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        foodList = self.getFood(successor).asList()
        features['successor'] = self.getScore(successor)

        # A heuristic algorithm to find the nearest food
        if len(foodList) > 0:
            min_distance = min([self.getMazeDistance(agentPosition, food) for food in foodList])
            features['foodDist'] = min_distance

        # A heuristic algorithm to pick up the food
        if len(foodList) > 0:
            features['pickupFood'] = -len(foodList) + 100 * self.getScore(successor)

        # Calculate the distance to enemy
        dist_enemy = self.enemy_distance(successor)
        if dist_enemy != None:
            if (dist_enemy <= 2):
                features['enemyClose'] = 4 / dist_enemy
            elif (dist_enemy <= 4):
                features['enemyClose'] = 1
            else:
                features['enemyClose'] = 0

        # Calculate distance to capsule
        capsules = self.getCapsules(successor)
        if (len(capsules) > 0):
            minCapsuleDist = min([self.getMazeDistance(agentPosition, capsule) for capsule in capsules])
            features['pickupCapsule'] = -len(capsules)
        else:
            minCapsuleDist = .1
        features['capsuleDist'] = 1.0 / minCapsuleDist

        # A heuristic algorithm to keep food
        if agentPosition in self.getFood(gameState).asList():
            self.foodNum += 1.0
        if self.side(gameState) == 0.0:
            self.foodNum = 0.0
        features['holdFood'] = self.foodNum * (min([self.distancer.getDistance(agentPosition, p) for p in
                                                    [(width / 2, i) for i in range(1, height) if
                                                     not gameState.hasWall(width / 2, i)]])) * self.side(gameState)

        # A heuristic algorithm to drop off food
        features['dropFood'] = self.foodNum * (self.side(gameState))

        # Create a power timer when meeting with a capsule
        if agentPosition in self.getCapsules(gameState):
            self.powerTimer = 120

        # Decrease the number of power timer when the timer is full
        if self.powerTimer > 0:
            self.powerTimer -= 1

        # Judge when the heuristic algorithm is powered
        if (self.isPowered()):
            features['isPowered'] = self.powerTimer / 120
            features['holdFood'] = 0.0
            features['pickupFood'] = 100 * features['pickupFood']
        else:
            features['isPowered'] = 0.0

        # Compute distance to partner
        if self.isAgentPacman(successor):
            dist_to_partner = self.get_dist_to_partner(successor)
            # distanceToAgent is always None for one of the agents (so they don't get stuck)
            if dist_to_partner != None:
                features['dist_to_partner'] = 1.0 / dist_to_partner

        # Judge whether the algorithm is dead or not
        actions = gameState.getLegalActions(self.index)
        if len(actions) <= 2:
            features['dead'] = 1.0
        else:
            features['dead'] = 0.0

        # Terminate the algorithm
        if (action == Directions.STOP):
            features['stop'] = 1.0
        else:
            features['stop'] = 0.0

        return features

    # Return all the start_features
    def start_feature(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        agentPosition = myState.getPosition()

        # Compute distance to board centre
        dist_to_center = self.getMazeDistance(agentPosition, self.center)
        features['wayToCenter'] = dist_to_center
        if agentPosition == self.center:
            features['atCenter'] = 1
        return features

    # Return the weight for attacking in the heuristic algorithm
    def attack_weight(self, gameState, action):
        return {'successor': 800, 'foodDist': -10, 'enemyClose': -1000,
                'pickupFood': 4000, 'capsuleDist': 700, 'stop': -1000, 'dead': -50,
                'isPowered': 999999, 'dropFood': 600, 'holdFood': -20,
                'distToAlly': -6000, 'pickupCapsule': 5000}

    # Return the weight for starting in the heuristic algorithm
    def start_weight(self, gameState, action):
        return {'wayToCenter': -1, 'atCenter': 1000}


class Attack(AttackAgent):

    def runAgent(self, gameState):
        locations = []
        self.atCenter = False
        x = gameState.getWalls().width / 2
        y = gameState.getWalls().height / 2
        if self.red:
            x = x - 1
        self.center = (x, y)
        maxHeight = gameState.getWalls().height
        for i in xrange(maxHeight - y):
            if not gameState.hasWall(x, y):
                locations.append((x, y))

        agentPosition = gameState.getAgentState(self.index).getPosition()
        minDist = 9999999
        minPos = None
        for location in locations:
            dist = self.getMazeDistance(agentPosition, location)
            if dist <= minDist:
                minDist = dist
                minPos = location

        self.center = minPos


class DefendAgent(CaptureAgent):

    # Initialize position and power timer for Agent
    def __init__(self, gameState):
        CaptureAgent.__init__(self, gameState)
        self.pos = [None] * 4

    def registerInitialState(self, gameState):

        CaptureAgent.registerInitialState(self, gameState)

        # Get team color
        if self.red:
            CaptureAgent.registerTeam(self, gameState.getRedTeamIndices())
        else:
            CaptureAgent.registerTeam(self, gameState.getBlueTeamIndices())

        # Get the board space
        self.x = gameState.data.layout.width
        self.y = gameState.data.layout.height

        # Get legal positions without walls
        self.legalPositions = []
        for p in gameState.getWalls().asList(False):
            if p[1] > 1:
                self.legalPositions.append(p)
        self.walls = list(gameState.getWalls())
        self.chokes = []

        if self.red:
            coolDistance = -3
        else:
            coolDistance = 4
        for i in range(self.y):
            if not self.walls[self.x / 2 + coolDistance][i]:
                self.chokes.append(((self.x / 2 + coolDistance), i))
        if self.index == max(gameState.getRedTeamIndices()) or self.index == max(gameState.getBlueTeamIndices()):
            x, y = self.chokes[3 * len(self.chokes) / 4]
        else:
            x, y = self.chokes[1 * len(self.chokes) / 4]
        self.goalSetting = (x, y)

        self.enemies = self.getOpponents(gameState)

        global beliefs
        beliefs = [util.Counter()] * gameState.getNumAgents()
        for i, val in enumerate(beliefs):
            if i in self.enemies:
                beliefs[i][gameState.getInitialAgentPosition(i)] = 1

        self.runAgent(gameState)

    def getEnemies(self, gameState):
        enemyPositions = []
        for enemy in self.enemies:
            position = gameState.getAgentPosition(enemy)
            if position != None:
                enemyPositions.append((enemy, position))
        return enemyPositions

    # Find the closest enemy
    def enemy_distance(self, gameState):
        enemyPositions = self.getEnemies(gameState)
        minDist = []
        if len(enemyPositions) > 0:
            agentPosition = self.getAgentPosition(gameState)
            for enemy, position in enemyPositions:
                minDist.append(self.getMazeDistance(position, agentPosition))
                minDist.sort()
            return minDist[0]
        else:
            return None

    def isAgentPacman(self, gameState):
        return gameState.getAgentState(self.index).isPacman

    def getAgentPosition(self, gameState):
        return gameState.getAgentPosition(self.index)

    def get_dist_to_partner(self, gameState):
        distanceToAgent = None
        agentsList = self.agentsOnTeam
        if self.index == agentsList[0]:
            otherAgentIndex = agentsList[1]
            distanceToAgent = None
        else:
            otherAgentIndex = agentsList[0]
            agentPosition = self.getAgentPosition(gameState)
            anotherPosition = gameState.getAgentState(otherAgentIndex).getPosition()
            distanceToAgent = self.getMazeDistance(agentPosition, anotherPosition)
            if distanceToAgent == 0:
                distanceToAgent = 0.1
        return distanceToAgent

    def side(self, gameState):
        width = gameState.data.layout.width
        pos = gameState.getAgentPosition(self.index)
        if self.index % 2 == 1:
            if pos[0] < width / (2):
                return 1.0
            else:
                return 0.0
        else:
            if pos[0] > width / 2 - 1:
                return 1.0
            else:
                return 0.0
    # Calculate how long the ghost is scared
    def ScaredTimer(self, gameState):
        return gameState.getAgentState(self.index).scaredTimer

    def getDist(self, p):
        posActions = [(p[0] - 1, p[1]), (p[0] + 1, p[1]), (p[0], p[1] - 1), (p[0], p[1] + 1),
                      (p[0], p[1])]  # go East,West,North,South and Stop
        actions = []
        for action in posActions:
            if action in self.legalPositions:
                actions.append(action)
        distance = util.Counter()
        for action in actions:
            distance[action] = 1
        return distance

    def time_elapsed(self, gameState):
        for agent, belief in enumerate(beliefs):
            if agent in self.enemies:
                new_beliefs = util.Counter()
                position = gameState.getAgentPosition(agent)
                if position != None:
                    new_beliefs[position] = 1.0
                else:
                    for p in belief:
                        if p in self.legalPositions and belief[p] > 0:
                            newPosDist = self.getDist(p)
                            for x, y in newPosDist:
                                new_beliefs[x, y] += belief[p] * newPosDist[x, y]
                    if len(new_beliefs) == 0:
                        old_state = self.getPreviousObservation()
                        if old_state is not None and old_state.getAgentPosition(agent) != None:  # ate an enemy
                            new_beliefs[old_state.getInitialAgentPosition(agent)] = 1.0
                        else:
                            for p in self.legalPositions: new_beliefs[p] = 1.0
                beliefs[agent] = new_beliefs

    # Search and observe the enemies
    def observe(self, agent, noisyDistance, gameState):
        agent_position = gameState.getAgentPosition(self.index)
        all_possible = util.Counter()
        for position in self.legalPositions:
            true_distance = util.manhattanDistance(position, agent_position)
            all_possible[position] += gameState.getDistanceProb(true_distance, noisyDistance)
            beliefs[agent][position] *= all_possible[position]

    # Get the best action based on different situations
    def chooseAction(self, gameState):
        enemies = self.enemies
        noisy_distance = gameState.getAgentDistances()

        for enemy in enemies:
            self.observe(enemy, noisy_distance[enemy], gameState)

        self.locations = [self.chokes[len(self.chokes) / 2]] * gameState.getNumAgents()
        for i, belief in enumerate(beliefs):
            max_loction = 0
            is_all_equal = 0
            for val in beliefs[i]:

                if belief[val] == max_loction and max_loction > 0:

                    is_all_equal += 1
                elif belief[val] > max_loction:
                    max_loction = belief[val]
                    self.locations[i] = val

            if is_all_equal > 5:
                self.locations[i] = self.goalSetting

        for enemy in enemies:
            beliefs[enemy].normalize()
            self.pos[enemy] = max(beliefs[enemy].iteritems(), key=operator.itemgetter(1))[0]

        # Next Step
        self.time_elapsed(gameState)
        agentPos = gameState.getAgentPosition(self.index)

        # Default to defend mode
        type_eval = 'defend'

        # Start in the start state before move to center
        if self.atCenter == False:
            type_eval = 'start'

        # If at centre, switch to defend
        if agentPos == self.center and self.atCenter == False:
            self.atCenter = True
            type_eval = 'defend'

        # If enemies are eating the pacman's food then starting hunting them
        for enemy in enemies:
            if (gameState.getAgentState(enemy).isPacman):
                type_eval = 'hunt'

        # If enemies appear in the pacman's side then starting defending
        enemyPositions = self.getEnemies(gameState)
        if len(enemyPositions) > 0:
            for enemy, pos in enemyPositions:
                if self.getMazeDistance(agentPos, pos) < 5 and not self.isAgentPacman(gameState):
                    type_eval = 'defend'
                    break

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

    def getSuccessor(self, gameState, action):
        """
    Find the next successor which is a grid position (location tuple).
    """
        successor = gameState.generateSuccessor(self.index, action)
        position = successor.getAgentState(self.index).getPosition()
        if position != nearestPoint(position):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    # Calculate heurisic score
    def evaluate(self, gameState, action, type_eval):
        if type_eval == 'defend':
            features = self.defend_feature(gameState, action)
            weights = self.defend_weight(gameState, action)
        elif type_eval == 'start':
            features = self.start_feature(gameState, action)
            weights = self.start_weight(gameState, action)
        elif type_eval == 'hunt':
            features = self.hunt_feature(gameState, action)
            weights = self.hunt_weight(gameState, action)

        return features * weights

    # Get the heuristic features when hunting
    def hunt_feature(self, gameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        agent_state = successor.getAgentState(self.index)
        agentPosition = agent_state.getPosition()
        opponents = self.getOpponents(gameState)
        invaders = [agent for agent in opponents if successor.getAgentState(agent).isPacman]
        features['invader_number'] = len(invaders)
        # change to commit
        enemy_distance = 99999
        
        for enemy in invaders:
            enemy_position = self.pos[enemy]
            enemy_distance = self.getMazeDistance(agentPosition, enemy_position)
        features['invader_distance'] = enemy_distance

        # Calculate the distance to partner of the pacman
        if self.isAgentPacman(successor):
            dist_to_partner = self.get_dist_to_partner(successor)
            if dist_to_partner != None:
                features['dist_to_partner'] = 1.0 / dist_to_partner

        if action == Directions.STOP: features['stop'] = 1
        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == reverse: features['reverse'] = 1

        return features

    # A heuristic algorithm for defending
    def defend_feature(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        agentPosition = myState.getPosition()
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [enemy for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]
        features['invader_number'] = len(invaders)
        if len(invaders) > 0:
            enemy_distance = [self.getMazeDistance(agentPosition, enemy.getPosition()) for enemy in invaders]
            features['invader_distance'] = min(enemy_distance)

        distEnemy = self.enemy_distance(successor)
        if (distEnemy <= 5):
            features['enemyClose'] = 1
            if (distEnemy <= 1 and self.ScaredTimer(successor) > 0):
                features['enemyClose'] = -1
        else:
            features['enemyClose'] = 0

        if self.isAgentPacman(successor):
            dist_to_partner = self.get_dist_to_partner(successor)
            if dist_to_partner != None:
                features['dist_to_partner'] = 1.0 / dist_to_partner

        if action == Directions.STOP: features['stop'] = 1
        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == reverse: features['reverse'] = 1

        return features

    # Get the heuristic features when starting
    def start_feature(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        agentPosition = myState.getPosition()
        dist = self.getMazeDistance(agentPosition, self.center)
        features['wayToCenter'] = dist
        if agentPosition == self.center:
            features['atCenter'] = 1
        return features

    # Get the weight of the heuristic features when hunting
    def hunt_weight(self, gameState, action):

        return {'invader_number': -100, 'invader_distance': -10, 'stop': -5000,
                'reverse': -5000, 'dist_to_partner': -2500}

    # Get the weight of the heuristic features when defending
    def defend_weight(self, gameState, action):
        return {'invader_number': -10000, 'invader_distance': -500, 'stop': -5000,
                'reverse': -200, 'enemyClose': 3000, 'dist_to_partner': -4000}

    # Get the weight of the heuristic features when starting
    def start_weight(self, gameState, action):
        return {'wayToCenter': -1, 'atCenter': 7000}


class Defend(DefendAgent):

    def runAgent(self, gameState):
        locations = []
        self.atCenter = False
        x = gameState.getWalls().width / 2
        y = gameState.getWalls().height / 2
        if self.red:
            x = x - 1
        self.center = (x, y)
        for i in xrange(y):
            if not gameState.hasWall(x, y):
                locations.append((x, y))

        agentPosition = gameState.getAgentState(self.index).getPosition()
        minDist = 9999999
        minPos = None
        for location in locations:
            dist = self.getMazeDistance(agentPosition, location)
            if dist <= minDist:
                minDist = dist
                minPos = location

        self.center = minPos
