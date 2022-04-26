import numpy as np
from math import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, random, time
from env import single_expert_dynamics,single_expert_stochastic_dynamics, expert1_reward, expert2_reward, expert3_reward, expert1_cost, expert2_cost, expert3_cost

def optimal_policy(Q_matrix,num_action):
  distribution=np.zeros((9,9,num_action))
  distribution=distribution.astype(np.object)
  for x in range(9):
    for y in range(9):
      counter=0
      value_list=[]
      for a in range(num_action):
        value_list.append(Q_matrix[x][y][a])
      max_value=max(value_list)
      for a in range(num_action):
        if Q_matrix[x][y][a]==max_value:
          counter=counter+1
      for a in range(num_action):
        if Q_matrix[x][y][a]==max_value:
          distribution[x][y][a]=1.0/counter
  return distribution

def Q1_matrix_function(gamma,V_matrix,num_action,reward1_matrix,cost1_matrix):
  Q_matrix=np.zeros((9,9,num_action))
  Q_matrix=Q_matrix.astype(np.object)
  for x in range(9):
    for y in range(9):
      for a in range(num_action):
        next_state=single_expert_dynamics(np.mat([x,y]).T,np.mat([a]).T)
        value=0.8*V_matrix[next_state.item(0)][next_state.item(1)]+0.2*V_matrix[x][y]
        Q_matrix[x][y][a]=reward1_matrix[x,y,a]-cost1_matrix[x,y,a]+gamma*value
  return Q_matrix

def Q2_matrix_function(gamma,V_matrix,num_action,reward2_matrix,cost2_matrix):
  Q_matrix=np.zeros((9,9,num_action))
  Q_matrix=Q_matrix.astype(np.object)
  for x in range(9):
    for y in range(9):
      for a in range(num_action):
        next_state=single_expert_dynamics(np.mat([x,y]).T,np.mat([a]).T)
        value=0.8*V_matrix[next_state.item(0)][next_state.item(1)]+0.2*V_matrix[x][y]
        Q_matrix[x][y][a]=reward2_matrix[x,y,a]-cost2_matrix[x,y,a]+gamma*value
  return Q_matrix

def Q3_matrix_function(gamma,V_matrix,num_action,reward3_matrix,cost3_matrix):
  Q_matrix=np.zeros((9,9,num_action))
  Q_matrix=Q_matrix.astype(np.object)
  for x in range(9):
    for y in range(9):
      for a in range(num_action):
        next_state=single_expert_dynamics(np.mat([x,y]).T,np.mat([a]).T)
        value=0.8*V_matrix[next_state.item(0)][next_state.item(1)]+0.2*V_matrix[x][y]
        Q_matrix[x][y][a]=reward3_matrix[x,y,a]-cost3_matrix[x,y,a]+gamma*value
  return Q_matrix

def V_matrix_funciton(Q_matrix,num_action,policy):
  V_matrix=np.zeros((9,9))
  V_matrix=V_matrix.astype(np.object)
  for x in range(9):
    for y in range(9):
      value=0.0
      for a in range(num_action):
        value=value+policy[x][y][a]*Q_matrix[x][y][a]
      V_matrix[x][y]=value
  return V_matrix

def calculate_optimal_policy(gamma,num_action,reward1_matrix,cost1_matrix,reward2_matrix,cost2_matrix,reward3_matrix,cost3_matrix):
  V1_matrix=np.zeros((9,9))
  V1_matrix=V1_matrix.astype(np.object)
  V2_matrix=np.zeros((9,9))
  V2_matrix=V2_matrix.astype(np.object)
  V3_matrix=np.zeros((9,9))
  V3_matrix=V3_matrix.astype(np.object)
  Q1_matrix=np.copy(Q1_matrix_function(gamma,V1_matrix,num_action,reward1_matrix,cost1_matrix))
  policy1=np.copy(optimal_policy(Q1_matrix,num_action))
  new_V1_matrix=np.copy(V_matrix_funciton(Q1_matrix,num_action,policy1))
  Q2_matrix=np.copy(Q2_matrix_function(gamma,V2_matrix,num_action,reward2_matrix,cost2_matrix))
  policy2=np.copy(optimal_policy(Q2_matrix,num_action))
  new_V2_matrix=np.copy(V_matrix_funciton(Q2_matrix,num_action,policy2))
  Q3_matrix=np.copy(Q3_matrix_function(gamma,V3_matrix,num_action,reward3_matrix,cost3_matrix))
  policy3=np.copy(optimal_policy(Q3_matrix,num_action))
  new_V3_matrix=np.copy(V_matrix_funciton(Q3_matrix,num_action,policy3))
  max_value1=0.0
  max_value2=0.0
  max_value3=0.0
  for x in range(9):
    for y in range(9):
      if max_value1<abs(V1_matrix[x][y]-new_V1_matrix[x][y]):
        max_value1=abs(V1_matrix[x][y]-new_V1_matrix[x][y])
      if max_value2<abs(V2_matrix[x][y]-new_V2_matrix[x][y]):
        max_value2=abs(V2_matrix[x][y]-new_V2_matrix[x][y])
      if max_value3<abs(V3_matrix[x][y]-new_V3_matrix[x][y]):
        max_value3=abs(V3_matrix[x][y]-new_V3_matrix[x][y])
  while max_value1>1.0 or max_value2>1.0 or max_value3>1.0:
    V1_matrix=np.copy(new_V1_matrix)
    Q1_matrix=np.copy(Q1_matrix_function(gamma,V1_matrix,num_action,reward1_matrix,cost1_matrix))
    policy1=np.copy(optimal_policy(Q1_matrix,num_action))
    new_V1_matrix=np.copy(V_matrix_funciton(Q1_matrix,num_action,policy1))

    V2_matrix=np.copy(new_V2_matrix)
    Q2_matrix=np.copy(Q2_matrix_function(gamma,V2_matrix,num_action,reward2_matrix,cost2_matrix))
    policy2=np.copy(optimal_policy(Q2_matrix,num_action))
    new_V2_matrix=np.copy(V_matrix_funciton(Q2_matrix,num_action,policy2))

    V3_matrix=np.copy(new_V3_matrix)
    Q3_matrix=np.copy(Q3_matrix_function(gamma,V3_matrix,num_action,reward3_matrix,cost3_matrix))
    policy3=np.copy(optimal_policy(Q3_matrix,num_action))
    new_V3_matrix=np.copy(V_matrix_funciton(Q3_matrix,num_action,policy3))

    max_value1=0.0
    max_value2=0.0
    max_value3=0.0
    for x in range(9):
      for y in range(9):
        if max_value1<abs(V1_matrix[x][y]-new_V1_matrix[x][y]):
          max_value1=abs(V1_matrix[x][y]-new_V1_matrix[x][y])
        if max_value2<abs(V2_matrix[x][y]-new_V2_matrix[x][y]):
          max_value2=abs(V2_matrix[x][y]-new_V2_matrix[x][y])
        if max_value3<abs(V3_matrix[x][y]-new_V3_matrix[x][y]):
          max_value3=abs(V3_matrix[x][y]-new_V3_matrix[x][y])
  return policy1, policy2, policy3

def optimal_trial(initial_state,policy1,policy2,policy3,num_action):
  trajectory=[]
  state=initial_state
  for i in range(30):
    policy1_distribution=policy1[state.item(0)][state.item(1)][:]
    choice1=[]
    for a in range(num_action):
      if policy1_distribution[a]>0.0:
        choice1.append(a)
    index1=np.random.randint(len(choice1))
    action1=choice1[index1]
    next_state1=single_expert_stochastic_dynamics(state[0:2],np.mat([action1]).T)
    policy2_distribution=policy2[state.item(2)][state.item(3)][:]
    choice2=[]
    for a in range(num_action):
      if policy2_distribution[a]>0.0:
        choice2.append(a)
    index2=np.random.randint(len(choice2))
    action2=choice2[index2]
    next_state2=single_expert_stochastic_dynamics(state[2:4],np.mat([action2]).T)
    policy3_distribution=policy3[state.item(4)][state.item(5)][:]
    choice3=[]
    for a in range(num_action):
      if policy3_distribution[a]>0.0:
        choice3.append(a)
    index3=np.random.randint(len(choice3))
    action3=choice3[index3]
    next_state3=single_expert_stochastic_dynamics(state[4:6],np.mat([action3]).T)
    trajectory.append([state.item(0),state.item(1),state.item(2),state.item(3),state.item(4),state.item(5),action1,action2,action3])
    state=np.copy(np.vstack((next_state1,next_state2,next_state3)))
  return trajectory

reward1_matrix=np.zeros((9,9,9))
reward2_matrix=np.zeros((9,9,9))
reward3_matrix=np.zeros((9,9,9))

reward1_matrix=reward1_matrix.astype(np.object)
reward2_matrix=reward2_matrix.astype(np.object)
reward3_matrix=reward3_matrix.astype(np.object)

cost1_matrix=np.zeros((9,9,9))
cost2_matrix=np.zeros((9,9,9))
cost3_matrix=np.zeros((9,9,9))

cost1_matrix=cost1_matrix.astype(np.object)
cost2_matrix=cost2_matrix.astype(np.object)
cost3_matrix=cost3_matrix.astype(np.object)

initial_state=np.mat([0,8,0,4,0,0]).T
num_action=9
gamma=0.9
omega=np.mat([1.0,-1.0,1.0,-1.0,1.0,-1.0]).T
theta1=np.ones((22,1))
theta1=theta1[:,np.newaxis]
theta1=np.mat(theta1).T
theta1[11:20]=0.0
theta2=np.ones((22,1))
theta2=theta2[:,np.newaxis]
theta2=np.mat(theta2).T
theta2[11:20]=0.0
theta3=np.ones((24,1))
theta3=theta3[:,np.newaxis]
theta3=np.mat(theta3).T
theta3[11:20]=0.0
theta3[22:24]=0.0
theta=np.vstack((theta1,theta2,theta3))

for x in range(9):
  for y in range(9):
    for a in range(9):
      reward1_matrix[x,y,a]=expert1_reward(omega[0:2],np.mat([x,y]).T,np.mat([a]).T)
      reward2_matrix[x,y,a]=expert2_reward(omega[2:4],np.mat([x,y]).T,np.mat([a]).T)
      reward3_matrix[x,y,a]=expert3_reward(omega[4:6],np.mat([x,y]).T,np.mat([a]).T)
      cost1_matrix[x,y,a]=expert1_cost(theta[0:22],np.mat([x,y]).T,np.mat([a]).T)
      cost2_matrix[x,y,a]=expert2_cost(theta[22:44],np.mat([x,y]).T,np.mat([a]).T)
      cost3_matrix[x,y,a]=expert3_cost(theta[44:68],np.mat([x,y]).T,np.mat([a]).T)


policy1,policy2,policy3=calculate_optimal_policy(gamma,num_action,reward1_matrix,cost1_matrix,reward2_matrix,cost2_matrix,reward3_matrix,cost3_matrix)
saved_policy1=policy1.reshape(9*9,9)
optimal_policy1_file=open("optimal_policy1_file.txt","w")
for entry in saved_policy1:
  np.savetxt(optimal_policy1_file,entry)
optimal_policy1_file.close()
saved_policy2=policy2.reshape(9*9,9)
optimal_policy2_file=open("optimal_policy2_file.txt","w")
for entry in saved_policy2:
  np.savetxt(optimal_policy2_file,entry)
optimal_policy2_file.close()
saved_policy3=policy3.reshape(9*9,9)
optimal_policy3_file=open("optimal_policy3_file.txt","w")
for entry in saved_policy3:
  np.savetxt(optimal_policy3_file,entry)
optimal_policy3_file.close()
trajectory=optimal_trial(initial_state,policy1,policy2,policy3,num_action)
print(trajectory)
    
  
  






























  

  


