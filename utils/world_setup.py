import numpy as np

def blueprintAgent(data):
    number_agents = data['Number of Agents']
    world_width = data['World Width']
    world_length = data['World Length']
    
    # Initialize all agents in the np.randomly in world
    data['Agent Positions BluePrint'] = np.random.rand(number_agents, 2) * [world_width, world_length]
    angleCol = np.random.uniform(-np.pi, np.pi, number_agents)
    data['Agent Orientations BluePrint'] = np.vstack((np.cos(angleCol), np.sin(angleCol))).T

def blueprintAgentInitSize(data):
    number_agents = data['Number of Agents']
    world_width = data['World Width']
    world_length = data['World Length']
    agentInitSize = data["Agent Initialization Size"]
    
    worldSize = np.array([world_width, world_length])
    
    # Initialize all agents in the np.randomly in world
    positionCol = np.random.rand(number_agents, 2) * worldSize
    positionCol *= agentInitSize
    positionCol += 0.5 * (1 - agentInitSize) * worldSize
    data['Agent Positions BluePrint'] = positionCol
    angleCol = np.random.uniform(-np.pi, np.pi, number_agents)
    data['Agent Orientations BluePrint'] = np.vstack((np.cos(angleCol), np.sin(angleCol))).T


    
    
def blueprintPoi(data):
    number_pois = data['Number of POIs']    
    world_width = data['World Width']
    world_length = data['World Length']  
    
    # Initialize all Pois np.randomly
    data['Poi Positions BluePrint'] = np.random.rand(number_pois, 2) * [world_width, world_length]
    data['Poi Values BluePrint'] = np.arange(number_pois) + 1.0
 
 
def initWorld(data):
    data['Agent Positions'] = data['Agent Positions BluePrint'].copy()
    data['Agent Orientations'] = data['Agent Orientations BluePrint'].copy()
    data['Poi Positions'] = data['Poi Positions BluePrint'].copy()
    data['Poi Values'] = data['Poi Values BluePrint'].copy()


def blueprintStatic(data):
    number_agents = data['Number of Agents']
    number_pois = data['Number of POIs'] 
    world_width = data['World Width']
    world_length = data['World Length']
    
    data['Agent Positions BluePrint'] = np.ones((number_agents,2)) * 0.5 * [world_width, world_length]
    angles = np.random.uniform(-np.pi, np.pi, number_agents)
    data['Agent Orientations BluePrint'] = np.vstack((np.cos(angles), np.sin(angles))).T
    data['Poi Positions BluePrint'] = data['Poi Relative Static Positions'] * [world_width, world_length]
    data['Poi Values BluePrint'] =  data['Poi Static Values'].copy()
    
def assignRandomPolicies(data):
    number_agents = data['Number of Agents']
    populationCol = data['Agent Populations']
    policyCol = [None] * number_agents
    for agentIndex in range(number_agents):
        policyCol[agentIndex] = np.random.choice(populationCol[agentIndex])
    data["Agent Policies"] = policyCol