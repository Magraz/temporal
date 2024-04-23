import numpy as np
cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def assignGlobalReward(data):
    
    cdef int number_agents = data['Number of Agents']
    cdef int number_pois = data['Number of POIs'] 
    cdef double minDistanceSqr = data["Minimum Distance"] ** 2
    cdef int historyStepCount = data["Steps"] + 1
    cdef int coupling = data["Coupling"]
    cdef double observationRadiusSqr = data["Observation Radius"] ** 2
    cdef double[:, :, :] agentPositionHistory = data["Agent Position History"]
    cdef double[:] poiValueCol = data['Poi Values']
    cdef double[:, :] poiPositionCol = data["Poi Positions"]
    cdef stepIndex = data["Step Index"]
    
    cdef int poiIndex, agentIndex, observerCount
    cdef double separation0, separation1, closestObsDistanceSqr, distanceSqr, stepClosestObsDistanceSqr
    cdef double Inf = float("inf")
    
    cdef double globalReward = 0.0
    cdef double scale = 0.0
    
    
    for poiIndex in range(number_pois):
        closestObsDistanceSqr = Inf
        #for stepIndex in range(historyStepCount):
        # Count how many agents observe poi, update closest distance if necessary
        observerCount = 0
        stepClosestObsDistanceSqr = 0
        for agentIndex in range(number_agents):
            # Calculate separation distance between poi and agent
            separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, agentIndex, 0]
            separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, agentIndex, 1]
            distanceSqr = separation0 * separation0 + separation1 * separation1
            
            # Check if agent observes poi, update closest step distance
            if distanceSqr < observationRadiusSqr:
                observerCount += 1
                if distanceSqr > stepClosestObsDistanceSqr:
                    stepClosestObsDistanceSqr += distanceSqr
                    
        # update closest distance only if poi is observed    
        if observerCount >= coupling:
            if stepClosestObsDistanceSqr < closestObsDistanceSqr:
                closestObsDistanceSqr = stepClosestObsDistanceSqr
        
        # add to global reward if poi is observed 
        if closestObsDistanceSqr < observationRadiusSqr:
            if closestObsDistanceSqr < minDistanceSqr:
                closestObsDistanceSqr = minDistanceSqr
            scale = (observationRadiusSqr-closestObsDistanceSqr) / (observationRadiusSqr - minDistanceSqr)
            globalReward += poiValueCol[poiIndex] * scale
    
    data["Global Reward"] = globalReward
    data["Agent Rewards"] = np.ones(number_agents) * globalReward


 
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def assignDifferenceReward(data):
    cdef int number_agents = data['Number of Agents']
    cdef int number_pois = data['Number of POIs'] 
    cdef double minDistanceSqr = data["Minimum Distance"] ** 2
    cdef int historyStepCount = data["Steps"] + 1
    cdef int coupling = data["Coupling"]
    cdef double observationRadiusSqr = data["Observation Radius"] ** 2
    cdef double[:, :, :] agentPositionHistory = data["Agent Position History"]
    cdef double[:] poiValueCol = data['Poi Values']
    cdef double[:, :] poiPositionCol = data["Poi Positions"]

    cdef int poiIndex, stepIndex, agentIndex, observerCount, otherAgentIndex
    cdef double separation0, separation1, closestObsDistanceSqr, distanceSqr, stepClosestObsDistanceSqr
    cdef double Inf = float("inf")
    
    cdef double globalReward = 0.0
    cdef double globalWithoutReward = 0.0
    cdef double scale = 0.0
    
    npDifferenceRewardCol = np.zeros(number_agents)
    cdef double[:] differenceRewardCol = npDifferenceRewardCol
    
    for poiIndex in range(number_pois):
        closestObsDistanceSqr = Inf
        for stepIndex in range(historyStepCount-1,historyStepCount):
            # Count how many agents observe poi, update closest distance if necessary
            observerCount = 0
            stepClosestObsDistanceSqr = Inf
            for agentIndex in range(number_agents):
                # Calculate separation distance between poi and agent
                separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, agentIndex, 0]
                separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, agentIndex, 1]
                distanceSqr = separation0 * separation0 + separation1 * separation1
                
                # Check if agent observes poi, update closest step distance
                if distanceSqr < observationRadiusSqr:
                    observerCount += 1
                    if distanceSqr < stepClosestObsDistanceSqr:
                        stepClosestObsDistanceSqr = distanceSqr
                        
                        
            # update closest distance only if poi is observed    
            if observerCount >= coupling:
                if stepClosestObsDistanceSqr < closestObsDistanceSqr:
                    closestObsDistanceSqr = stepClosestObsDistanceSqr
        
        # add to global reward if poi is observed 
        if closestObsDistanceSqr < observationRadiusSqr:
            if closestObsDistanceSqr < minDistanceSqr:
                closestObsDistanceSqr = minDistanceSqr
            scale = (observationRadiusSqr-closestObsDistanceSqr) / (observationRadiusSqr - minDistanceSqr)
            #print("scale",scale,poiValueCol[poiIndex],poiIndex )
            globalReward += poiValueCol[poiIndex] *scale

    
    for agentIndex in range(number_agents):
        globalWithoutReward = 0
        for poiIndex in range(number_pois):
            closestObsDistanceSqr = Inf
            for stepIndex in range(historyStepCount-1,historyStepCount):
                # Count how many agents observe poi, update closest distance if necessary
                observerCount = 0
                stepClosestObsDistanceSqr = Inf
                for otherAgentIndex in range(number_agents):
                    if agentIndex != otherAgentIndex:
                        # Calculate separation distance between poi and agent\
                        separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, otherAgentIndex, 0]
                        separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, otherAgentIndex, 1]
                        distanceSqr = separation0 * separation0 + separation1 * separation1
                        
                        # Check if agent observes poi, update closest step distance
                        if distanceSqr < observationRadiusSqr:
                            observerCount += 1
                            if distanceSqr < stepClosestObsDistanceSqr:
                                stepClosestObsDistanceSqr = distanceSqr
                            
                            
                # update closest distance only if poi is observed    
                if observerCount >= coupling:
                    if stepClosestObsDistanceSqr < closestObsDistanceSqr:
                        closestObsDistanceSqr = stepClosestObsDistanceSqr
            
            # add to global reward if poi is observed 
            if closestObsDistanceSqr < observationRadiusSqr:
                if closestObsDistanceSqr < minDistanceSqr:
                    closestObsDistanceSqr = minDistanceSqr
                scale = (observationRadiusSqr-closestObsDistanceSqr) / (observationRadiusSqr - minDistanceSqr)
                globalWithoutReward += poiValueCol[poiIndex] * scale
        differenceRewardCol[agentIndex] = globalReward - globalWithoutReward
        
    data["Agent Rewards"] = npDifferenceRewardCol  
    data["Global Reward"] = globalReward
    #print(globalReward)

 
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def assignD2(data):
    cdef int number_agents = data['Number of Agents']
    cdef int number_pois = data['Number of POIs'] 
    cdef double minDistanceSqr = data["Minimum Distance"] ** 2
    cdef int historyStepCount = data["Steps"] + 1
    cdef int coupling = data["Coupling"]
    cdef double observationRadiusSqr = data["Observation Radius"] ** 2
    cdef double[:, :, :] agentPositionHistory = data["Agent Position History"]
    cdef double[:] poiValueCol = data['Poi Values']
    cdef double[:, :] poiPositionCol = data["Poi Positions"]

    cdef int poiIndex, stepIndex, agentIndex, observerCount, otherAgentIndex
    cdef double separation0, separation1, closestObsDistanceSqr, distanceSqr, stepClosestObsDistanceSqr
    cdef double Inf = float("inf")
    
    cdef double globalReward = 0.0
    cdef double globalWithoutReward = 0.0
    cdef double scale = 0.0
    
    npDifferenceRewardCol = np.zeros((number_agents,number_agents))
    localR=np.zeros((number_agents,number_pois))
    cdef double[:,:] differenceRewardCol = npDifferenceRewardCol
    
    for poiIndex in range(number_pois):
        closestObsDistanceSqr = Inf
        for stepIndex in range(historyStepCount-1,historyStepCount):
            # Count how many agents observe poi, update closest distance if necessary
            observerCount = 0
            stepClosestObsDistanceSqr = Inf
            for agentIndex in range(number_agents):
                # Calculate separation distance between poi and agent
                separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, agentIndex, 0]
                separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, agentIndex, 1]
                distanceSqr = separation0 * separation0 + separation1 * separation1
                localR[agentIndex,poiIndex]=-distanceSqr
                # Check if agent observes poi, update closest step distance
                if distanceSqr < observationRadiusSqr:
                    observerCount += 1
                    if distanceSqr < stepClosestObsDistanceSqr:
                        stepClosestObsDistanceSqr = distanceSqr
                        
                        
            # update closest distance only if poi is observed    
            if observerCount >= coupling:
                if stepClosestObsDistanceSqr < closestObsDistanceSqr:
                    closestObsDistanceSqr = stepClosestObsDistanceSqr
        
        # add to global reward if poi is observed 
        if closestObsDistanceSqr < observationRadiusSqr:
            if closestObsDistanceSqr < minDistanceSqr:
                closestObsDistanceSqr = minDistanceSqr
            scale = (observationRadiusSqr-closestObsDistanceSqr) / (observationRadiusSqr - minDistanceSqr)
            globalReward += poiValueCol[poiIndex] *scale

    
    for agentIndex1 in range(number_agents):
        for agentIndex2 in range(number_agents):
            globalWithoutReward = 0
            for poiIndex in range(number_pois):
                closestObsDistanceSqr = Inf
                for stepIndex in range(historyStepCount-1,historyStepCount):
                    # Count how many agents observe poi, update closest distance if necessary
                    observerCount = 0
                    stepClosestObsDistanceSqr = Inf
                    for otherAgentIndex in range(number_agents):
                        if agentIndex1 != otherAgentIndex and agentIndex2 != otherAgentIndex:
                            # Calculate separation distance between poi and agent\
                            separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, otherAgentIndex, 0]
                            separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, otherAgentIndex, 1]
                            distanceSqr = separation0 * separation0 + separation1 * separation1
                            
                            # Check if agent observes poi, update closest step distance
                            if distanceSqr < observationRadiusSqr:
                                observerCount += 1
                                if distanceSqr < stepClosestObsDistanceSqr:
                                    stepClosestObsDistanceSqr = distanceSqr
                                
                                
                    # update closest distance only if poi is observed    
                    if observerCount >= coupling:
                        if stepClosestObsDistanceSqr < closestObsDistanceSqr:
                            closestObsDistanceSqr = stepClosestObsDistanceSqr
                
                # add to global reward if poi is observed 
                if closestObsDistanceSqr < observationRadiusSqr:
                    if closestObsDistanceSqr < minDistanceSqr:
                        closestObsDistanceSqr = minDistanceSqr
                    scale = (observationRadiusSqr-closestObsDistanceSqr) / (observationRadiusSqr - minDistanceSqr)
                    globalWithoutReward += poiValueCol[poiIndex] * scale
            differenceRewardCol[agentIndex1,agentIndex2] = globalReward - globalWithoutReward
            differenceRewardCol[agentIndex2,agentIndex1] = globalReward - globalWithoutReward

    data["Local Rewards"] = localR    
    data["Agent Rewards"] = npDifferenceRewardCol  
    data["Global Reward"] = globalReward
    



