import numpy as np
from math import *
from env import single_expert_dynamics,single_expert_stochastic_dynamics, expert1_reward, expert2_reward, expert3_reward, expert1_cost, expert2_cost, expert3_cost, feature1, feature2, feature3, expert_1_basis_constraint, expert_2_basis_constraint, expert_3_basis_constraint
from multiprocessing import Process

def empirical_feature_counts(trajectories,num_data):
  counts=0
  for i in range(num_data):
    one_count=0
    single_trajectory=trajectories[30*i:30*(i+1),:]
    for j in range(30):
      state=np.mat(np.copy(single_trajectory[j][0:6])).T
      action=np.mat(np.copy(single_trajectory[j][6:9])).T
      expert1_feature=feature1(state[0:2],action[0])
      expert2_feature=feature2(state[2:4],action[1])
      expert3_feature=feature3(state[4:6],action[2])
      one_count=one_count+np.vstack((expert1_feature,expert2_feature,expert3_feature))
    counts=counts+one_count
  return counts/num_data

def empirical_cost_counts(trajectories,num_data,theta):
  theta1=theta[0:22]
  theta2=theta[22:44]
  theta3=theta[44:68]
  counts=0
  for i in range(num_data):
    one_count=0
    single_trajectory=trajectories[30*i:30*(i+1),:]
    for j in range(30):
      state=np.mat(np.copy(single_trajectory[j][0:6])).T
      action=np.mat(np.copy(single_trajectory[j][6:9])).T
      cost=expert1_cost(theta1,state[0:2],action[0])+expert2_cost(theta2,state[2:4],action[1])+expert3_cost(theta3,state[4:6],action[2])
      one_count=one_count+cost
    counts=counts+one_count
  return counts/num_data

def empirical_constraint_counts(trajectories,num_data,lam):
  counts=0
  for i in range(num_data):
    one_count=0
    single_trajectory=trajectories[30*i:30*(i+1),:]
    for j in range(30):
      state=np.mat(np.copy(single_trajectory[j][0:6])).T
      action=np.mat(np.copy(single_trajectory[j][6:9])).T
      constraint1=expert_1_basis_constraint(state[0:2],action[0])
      constraint2=expert_2_basis_constraint(state[2:4],action[1])
      constraint3=expert_3_basis_constraint(state[4:6],action[2])
      constraint=np.vstack((constraint1,constraint2,constraint3))
      one_count=one_count+lam*constraint
    counts=counts+one_count
  return counts/num_data


def choose_action(policy_distribution):  # distribution is 9x1
  choice=np.random.uniform()
  sum_value=0.0
  for a in range(num_action):
    sum_value=sum_value+policy_distribution[a]
    if sum_value>=choice:
      return a

def trial(initial_state,policy1,policy2,policy3,num_action):
  trajectory=[]
  state=initial_state
  for i in range(30):
    policy1_distribution=policy1[state.item(0)][state.item(1)][:]
    action1=choose_action(policy1_distribution)
    next_state1=single_expert_stochastic_dynamics(state[0:2],np.mat([action1]).T)
    policy2_distribution=policy2[state.item(2)][state.item(3)][:]
    action2=choose_action(policy2_distribution)
    next_state2=single_expert_stochastic_dynamics(state[2:4],np.mat([action2]).T)
    policy3_distribution=policy3[state.item(4)][state.item(5)][:]
    action3=choose_action(policy3_distribution)
    next_state3=single_expert_stochastic_dynamics(state[4:6],np.mat([action3]).T)
    trajectory.append([state.item(0),state.item(1),state.item(2),state.item(3),state.item(4),state.item(5),action1,action2,action3])
    state=np.copy(np.vstack((next_state1,next_state2,next_state3)))
  return trajectory

def soft_policy(Q_matrix,V_matrix,num_action):
  distribution=np.zeros((9,9,num_action))
  distribution=distribution.astype(np.object)
  for x in range(9):
    for y in range(9):
      for a in range(num_action):
        distribution[x][y][a]=exp(Q_matrix[x][y][a])/exp(V_matrix[x][y])
  return distribution

def soft_Q_matrix_function(gamma,reward_matrix,lam,cost_matrix,V_matrix,num_action):
  Q_matrix=np.zeros((9,9,num_action))
  Q_matrix=Q_matrix.astype(np.object)
  for x in range(9):
    for y in range(9):
      for a in range(num_action):
        next_state=single_expert_dynamics(np.mat([x,y]).T,np.mat([a]).T)
        value=0.8*V_matrix[next_state.item(0)][next_state.item(1)]+0.2*V_matrix[x][y]
        Q_matrix[x][y][a]=reward_matrix[x,y,a]-lam*cost_matrix[x,y,a]+gamma*value
  return Q_matrix

  
def soft_V_matrix_funciton(Q_matrix,num_action):
  V_matrix=np.zeros((9,9))
  V_matrix=V_matrix.astype(np.object)
  for x in range(9):
    for y in range(9):
      value=0.0
      for a in range(num_action):
        value=value+exp(Q_matrix[x][y][a])
      V_matrix[x][y]=log(value)
  return V_matrix

def calculate_soft_policy(omega,theta,lam,gamma,num_action):

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

  for x in range(9):
    for y in range(9):
      for a in range(9):
        reward1_matrix[x,y,a]=expert1_reward(omega[0:2],np.mat([x,y]).T,np.mat([a]).T)
        reward2_matrix[x,y,a]=expert2_reward(omega[2:4],np.mat([x,y]).T,np.mat([a]).T)
        reward3_matrix[x,y,a]=expert3_reward(omega[4:6],np.mat([x,y]).T,np.mat([a]).T)
        cost1_matrix[x,y,a]=expert1_cost(theta[0:22],np.mat([x,y]).T,np.mat([a]).T)
        cost2_matrix[x,y,a]=expert2_cost(theta[22:44],np.mat([x,y]).T,np.mat([a]).T)
        cost3_matrix[x,y,a]=expert3_cost(theta[44:68],np.mat([x,y]).T,np.mat([a]).T)

  soft_V1_matrix=np.zeros((9,9))
  soft_V1_matrix=soft_V1_matrix.astype(np.object)
  soft_Q1_matrix=np.copy(soft_Q_matrix_function(gamma,reward1_matrix,lam,cost1_matrix,soft_V1_matrix,num_action))
  new_soft_V1_matrix=np.copy(soft_V_matrix_funciton(soft_Q1_matrix,num_action))
  soft_V2_matrix=np.zeros((9,9))
  soft_V2_matrix=soft_V2_matrix.astype(np.object)
  soft_Q2_matrix=np.copy(soft_Q_matrix_function(gamma,reward2_matrix,lam,cost2_matrix,soft_V2_matrix,num_action))
  new_soft_V2_matrix=np.copy(soft_V_matrix_funciton(soft_Q2_matrix,num_action))
  soft_V3_matrix=np.zeros((9,9))
  soft_V3_matrix=soft_V3_matrix.astype(np.object)
  soft_Q3_matrix=np.copy(soft_Q_matrix_function(gamma,reward3_matrix,lam,cost3_matrix,soft_V3_matrix,num_action))
  new_soft_V3_matrix=np.copy(soft_V_matrix_funciton(soft_Q3_matrix,num_action))
  #saved_Q_matrix=soft_Q_matrix.reshape(9*9*9,9*9*9)
  #soft_Q_file=open("soft_Q_file.txt","w")
  #for entry in saved_Q_matrix:
  #  np.savetxt(soft_Q_file,entry)
  #soft_Q_file.close()
  #saved_V_matrix=new_soft_V_matrix.reshape(9*9,9*9)
  #soft_V_file=open("soft_V_file.txt","w")
  #for entry in saved_V_matrix:
  #  np.savetxt(soft_V_file,entry)
  #soft_V_file.close()

  #load_soft_V_matrix=np.loadtxt("soft_V_file.txt",dtype=float)
  #new_soft_V_matrix=load_soft_V_matrix.reshape(9,9,9,9)
  #new_soft_V_matrix=new_soft_V_matrix.astype(np.object)
  max_value1=0.0
  max_value2=0.0
  max_value3=0.0
  for x in range(9):
    for y in range(9):
      if max_value1<abs(soft_V1_matrix[x][y]-new_soft_V1_matrix[x][y]):
        max_value1=abs(soft_V1_matrix[x][y]-new_soft_V1_matrix[x][y])
      if max_value2<abs(soft_V2_matrix[x][y]-new_soft_V2_matrix[x][y]):
        max_value2=abs(soft_V2_matrix[x][y]-new_soft_V2_matrix[x][y])
      if max_value3<abs(soft_V3_matrix[x][y]-new_soft_V3_matrix[x][y]):
        max_value3=abs(soft_V3_matrix[x][y]-new_soft_V3_matrix[x][y])
  while max_value1>1.0 or max_value2>1.0 or max_value3>1.0:
    #print(max_value3)
    soft_V1_matrix=np.copy(new_soft_V1_matrix)
    soft_Q1_matrix=np.copy(soft_Q_matrix_function(gamma,reward1_matrix,lam,cost1_matrix,soft_V1_matrix,num_action))
    new_soft_V1_matrix=np.copy(soft_V_matrix_funciton(soft_Q1_matrix,num_action))
    soft_V2_matrix=np.copy(new_soft_V2_matrix)
    soft_Q2_matrix=np.copy(soft_Q_matrix_function(gamma,reward2_matrix,lam,cost2_matrix,soft_V2_matrix,num_action))
    new_soft_V2_matrix=np.copy(soft_V_matrix_funciton(soft_Q2_matrix,num_action))
    soft_V3_matrix=np.copy(new_soft_V3_matrix)
    soft_Q3_matrix=np.copy(soft_Q_matrix_function(gamma,reward3_matrix,lam,cost3_matrix,soft_V3_matrix,num_action))
    new_soft_V3_matrix=np.copy(soft_V_matrix_funciton(soft_Q3_matrix,num_action))
    max_value1=0.0
    max_value2=0.0
    max_value3=0.0
    for x in range(9):
      for y in range(9):
        if max_value1<abs(soft_V1_matrix[x][y]-new_soft_V1_matrix[x][y]):
          max_value1=abs(soft_V1_matrix[x][y]-new_soft_V1_matrix[x][y])
        if max_value2<abs(soft_V2_matrix[x][y]-new_soft_V2_matrix[x][y]):
          max_value2=abs(soft_V2_matrix[x][y]-new_soft_V2_matrix[x][y])
        if max_value3<abs(soft_V3_matrix[x][y]-new_soft_V3_matrix[x][y]):
          max_value3=abs(soft_V3_matrix[x][y]-new_soft_V3_matrix[x][y])
  policy1=np.copy(soft_policy(soft_Q1_matrix,new_soft_V1_matrix,num_action))
  policy2=np.copy(soft_policy(soft_Q2_matrix,new_soft_V2_matrix,num_action))
  policy3=np.copy(soft_policy(soft_Q3_matrix,new_soft_V3_matrix,num_action))
  return policy1, policy2, policy3

def feature_expectation(policy1,policy2,policy3,initial_state,num_trials):
  trajectories=np.zeros((0,9))
  for i in range(num_trials):
    trajectory=np.copy(trial(initial_state,policy1,policy2,policy3,num_action))
    trajectory_array=np.zeros((30,9))
    for j in range(30):
     trajectory_array[j,:]=np.copy(trajectory[j])
    trajectories=np.vstack((trajectories,trajectory_array))
  return empirical_feature_counts(trajectories,num_trials)

def cost_expectation(policy1,policy2,policy3,initial_state,num_trials,theta):
  trajectories=np.zeros((0,9))
  for i in range(num_trials):
    trajectory=np.copy(trial(initial_state,policy1,policy2,policy3,num_action))
    trajectory_array=np.zeros((30,9))
    for j in range(30):
     trajectory_array[j,:]=np.copy(trajectory[j])
    trajectories=np.vstack((trajectories,trajectory_array))
  return empirical_cost_counts(trajectories,num_trials,theta)

def reward_cost_list(trajectories,num_data):
  omega1=np.mat([1.0,-1.0]).T
  omega2=np.mat([1.0,-1.0]).T
  omega3=np.mat([1.0,-1.0]).T
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
  reward_list=[]
  cost_list=[]
  for i in range(num_data):
    reward=0.0
    cost=0.0
    single_trajectory=trajectories[30*i:30*(i+1),:]
    for j in range(30):
      x1=single_trajectory[j][0]
      y1=single_trajectory[j][1]
      a1=single_trajectory[j][6]
      if x1==4 and y1>=0 and y1<=3:
        single_trajectory[j+1:30,0]=0
        single_trajectory[j+1:30,1]=8
        single_trajectory[j:30,6]=0
        break
      if x1==3 and y1==4:
        single_trajectory[j+1:30,0]=0
        single_trajectory[j+1:30,1]=8
        single_trajectory[j:30,6]=0
        break
      if x1==2 and y1==5:
        single_trajectory[j+1:30,0]=0
        single_trajectory[j+1:30,1]=8
        single_trajectory[j:30,6]=0
        break
      if x1==5 and y1==4:
        single_trajectory[j+1:30,0]=0
        single_trajectory[j+1:30,1]=8
        single_trajectory[j:30,6]=0
        break
      if x1==6 and y1==5:
        single_trajectory[j+1:30,0]=0
        single_trajectory[j+1:30,1]=8
        single_trajectory[j:30,6]=0
        break
      if x1==4 and y1>=6 and y1<=8:
        single_trajectory[j+1:30,0]=0
        single_trajectory[j+1:30,1]=8
        single_trajectory[j:30,6]=0
        break
      if a1==7 or a1==8:
        single_trajectory[j+1:30,0]=0
        single_trajectory[j+1:30,1]=8
        single_trajectory[j+1:30,6]=0
        break

    for j in range(30):
      x2=single_trajectory[j][2]
      y2=single_trajectory[j][3]
      a2=single_trajectory[j][7]
      if x2==4 and y2>=0 and y2<=3:
        single_trajectory[j+1:30,2]=0
        single_trajectory[j+1:30,3]=4
        single_trajectory[j:30,7]=0
        break  
      if x2==3 and y2==4:
        single_trajectory[j+1:30,2]=0
        single_trajectory[j+1:30,3]=4
        single_trajectory[j:30,7]=0
        break
      if x2==2 and y2==5:
        single_trajectory[j+1:30,2]=0
        single_trajectory[j+1:30,3]=4
        single_trajectory[j:30,7]=0
        break
      if x2==5 and y2==4:
        single_trajectory[j+1:30,2]=0
        single_trajectory[j+1:30,3]=4
        single_trajectory[j:30,7]=0
        break
      if x2==6 and y2==5:
        single_trajectory[j+1:30,2]=0
        single_trajectory[j+1:30,3]=4
        single_trajectory[j:30,7]=0
        break
      if x2==4 and y2>=6 and y2<=8:
        single_trajectory[j+1:30,2]=0
        single_trajectory[j+1:30,3]=4
        single_trajectory[j:30,7]=0
        break
      if a2==5 or a2==8:
        single_trajectory[j+1:30,2]=0
        single_trajectory[j+1:30,3]=4
        single_trajectory[j+1:30,7]=0
        break

    for j in range(30):
      x3=single_trajectory[j][4]
      y3=single_trajectory[j][5]
      a3=single_trajectory[j][8]
      if x3==4 and y3>=0 and y3<=3:
        single_trajectory[j+1:30,4]=0
        single_trajectory[j+1:30,5]=0
        single_trajectory[j:30,8]=0
        break  
      if x3==3 and y3==4:
        single_trajectory[j+1:30,4]=0
        single_trajectory[j+1:30,5]=0
        single_trajectory[j:30,8]=0
        break
      if x3==2 and y3==5:
        single_trajectory[j+1:30,4]=0
        single_trajectory[j+1:30,5]=0
        single_trajectory[j:30,8]=0
        break
      if x3==5 and y3==4:
        single_trajectory[j+1:30,4]=0
        single_trajectory[j+1:30,5]=0
        single_trajectory[j:30,8]=0
        break
      if x3==6 and y3==5:
        single_trajectory[j+1:30,4]=0
        single_trajectory[j+1:30,5]=0
        single_trajectory[j:30,8]=0
        break
      if x3==4 and y3>=6 and y3<=8:
        single_trajectory[j+1:30,4]=0
        single_trajectory[j+1:30,5]=0
        single_trajectory[j:30,8]=0
        break
      if a3==5 or a3==6:
        single_trajectory[j+1:30,4]=0
        single_trajectory[j+1:30,5]=0
        single_trajectory[j+1:30,8]=0
        break
 
    for j in range(30):
      state1=np.mat(np.copy(single_trajectory[j][0:2])).T
      state2=np.mat(np.copy(single_trajectory[j][2:4])).T
      state3=np.mat(np.copy(single_trajectory[j][4:6])).T
      action1=np.mat(np.copy(single_trajectory[j][6])).T
      action2=np.mat(np.copy(single_trajectory[j][7])).T
      action3=np.mat(np.copy(single_trajectory[j][8])).T
      single_reward=expert1_reward(omega1,state1,action1)+expert2_reward(omega2,state2,action2)+expert3_reward(omega3,state3,action3)
      single_cost=expert1_cost(theta1,state1,action1)+expert2_cost(theta2,state2,action2)+expert3_cost(theta3,state3,action3)
      reward=reward+single_reward
      cost=cost+single_cost
    reward_list.append(reward)
    cost_list.append(cost)
  return reward_list, cost_list
  

def feature_cost_mean_sd(policy1,policy2,policy3,initial_state,num_trials,theta):
  trajectories=np.zeros((0,9))
  for i in range(num_trials):
    trajectory=np.copy(trial(initial_state,policy1,policy2,policy3,num_action))
    trajectory_array=np.zeros((30,9))
    for j in range(30):
     trajectory_array[j,:]=np.copy(trajectory[j])
    trajectories=np.vstack((trajectories,trajectory_array))
  feature_expectation=empirical_feature_counts(trajectories,num_trials)
  reward_list, cost_list=reward_cost_list(trajectories,num_trials)
  reward_mean=sum(reward_list)/len(reward_list)
  reward_sd=sqrt(np.var(reward_list))
  cost_mean=sum(cost_list)/len(cost_list)
  cost_sd=sqrt(np.var(cost_list))
  return feature_expectation, reward_mean, reward_sd, cost_mean, cost_sd

def KL(policy1,policy2,policy3,expert_policy1,expert_policy2,expert_policy3):
  divergence=0
  for x in range(9):
    for y in range(9):
      for a in range(9):
        if expert_policy1[x,y,a]!=0 and policy1[x,y,a]!=0:
          divergence=divergence+expert_policy1[x,y,a]*log(expert_policy1[x,y,a]/policy1[x,y,a])
        if expert_policy2[x,y,a]!=0 and policy2[x,y,a]!=0:
          divergence=divergence+expert_policy2[x,y,a]*log(expert_policy2[x,y,a]/policy2[x,y,a])
        if expert_policy3[x,y,a]!=0 and policy3[x,y,a]!=0:
          divergence=divergence+expert_policy3[x,y,a]*log(expert_policy3[x,y,a]/policy3[x,y,a])
  return divergence/(9*9*3)

def constraint_expectation(policy1,policy2,policy3,initial_state,num_trials,lam):
  trajectories=np.zeros((0,9))
  for i in range(num_trials):
    trajectory=np.copy(trial(initial_state,policy1,policy2,policy3,num_action))
    trajectory_array=np.zeros((30,9))
    for j in range(30):
     trajectory_array[j,:]=np.copy(trajectory[j])
    trajectories=np.vstack((trajectories,trajectory_array))
  return empirical_constraint_counts(trajectories,num_trials,lam)

def inner_loop(initial_state,omega1,omega2,omega3,omega4,lam1,lam2,lam3,lam4,theta,gamma,num_action,num_trials,cost_empirical1,cost_empirical2,cost_empirical3,cost_empirical4, feature_empirical1,feature_empirical2,feature_empirical3,feature_empirical4,num_data1,num_data2,num_data3,num_data4):
  i=0
  iterations=10
  while i<iterations:
    print('iteration={}' .format(i))
    policy1=calculate_soft_policy(omega1,theta,lam1,gamma,num_action)
    policy2=calculate_soft_policy(omega2,theta,lam2,gamma,num_action)
    policy3=calculate_soft_policy(omega3,theta,lam3,gamma,num_action)
    policy4=calculate_soft_policy(omega4,theta,lam4,gamma,num_action)
    feature1=feature_expectation(policy1,initial_state,num_trials)
    feature2=feature_expectation(policy2,initial_state,num_trials)
    feature3=feature_expectation(policy3,initial_state,num_trials)
    feature4=feature_expectation(policy4,initial_state,num_trials)
    cost1=cost_expectation(policy1,initial_state,num_trials,theta)
    cost2=cost_expectation(policy2,initial_state,num_trials,theta)
    cost3=cost_expectation(policy3,initial_state,num_trials,theta)
    cost4=cost_expectation(policy4,initial_state,num_trials,theta)
    feature_gradient1=feature1-feature_empirical1
    feature_gradient2=feature2-feature_empirical2
    feature_gradient3=feature3-feature_empirical3
    feature_gradient4=feature4-feature_empirical4
    cost_gradient1=cost1-cost_empirical1
    cost_gradient2=cost2-cost_empirical2
    cost_gradient3=cost3-cost_empirical3
    cost_gradient4=cost4-cost_empirical4

    print('feature_gradient1 is {}' .format(feature_gradient1))
    print('feature_gradient2 is {}' .format(feature_gradient2))
    print('feature_gradient3 is {}' .format(feature_gradient3))
    print('feature_gradient4 is {}' .format(feature_gradient4))

    print('cost_gradient1 is {}' .format(cost_gradient1))
    print('cost_gradient2 is {}' .format(cost_gradient2))
    print('cost_gradient3 is {}' .format(cost_gradient3))
    print('cost_gradient4 is {}' .format(cost_gradient4))

    if i%2==1:
      omega1=0.5*(omega1+omega2)-(1.0/12000)*num_data1*feature_gradient1
      omega2=0.5*(omega1+omega2)-(1.0/12000)*num_data2*feature_gradient2
      omega3=0.5*(omega3+omega4)-(1.0/12000)*num_data3*feature_gradient3
      omega4=0.5*(omega3+omega4)-(1.0/12000)*num_data4*feature_gradient4
      lam1=0.5*(lam1+lam2)-(1.0/12000)*num_data1*cost_gradient1
      lam2=0.5*(lam1+lam2)-(1.0/12000)*num_data2*cost_gradient2
      lam3=0.5*(lam3+lam4)-(1.0/12000)*num_data3*cost_gradient3
      lam4=0.5*(lam3+lam4)-(1.0/12000)*num_data4*cost_gradient4

    else:
      omega1=0.5*(omega1+omega4)-(1.0/12000)*num_data1*feature_gradient1
      omega2=0.5*(omega2+omega3)-(1.0/12000)*num_data2*feature_gradient2
      omega3=0.5*(omega2+omega3)-(1.0/12000)*num_data3*feature_gradient3
      omega4=0.5*(omega1+omega4)-(1.0/12000)*num_data4*feature_gradient4
      lam1=0.5*(lam1+lam4)-(1.0/12000)*num_data1*cost_gradient1
      lam2=0.5*(lam2+lam3)-(1.0/12000)*num_data2*cost_gradient2
      lam3=0.5*(lam2+lam3)-(1.0/12000)*num_data3*cost_gradient3
      lam4=0.5*(lam1+lam4)-(1.0/12000)*num_data4*cost_gradient4

    print('omega1 is {}' .format(omega1))
    print('omega2 is {}' .format(omega2))
    print('omega3 is {}' .format(omega3))
    print('omega4 is {}' .format(omega4))

    print('lam1 is {}' .format(lam1))
    print('lam2 is {}' .format(lam2))
    print('lam3 is {}' .format(lam3))
    print('lam4 is {}' .format(lam4))

  i=i+1

def last_interation_inner_loop(initial_state,omega,lam,theta,gamma,num_action,num_trials,cost_empirical,feature_empirical):
  distribution1=np.loadtxt("optimal_policy1_file.txt",dtype=float)
  expert_policy1=distribution1.reshape(9,9,num_action)
  distribution2=np.loadtxt("optimal_policy2_file.txt",dtype=float)
  expert_policy2=distribution2.reshape(9,9,num_action)
  distribution3=np.loadtxt("optimal_policy3_file.txt",dtype=float)
  expert_policy3=distribution3.reshape(9,9,num_action)

  i=0
  iterations=121
  
  divergence_list=np.zeros((iterations,1))
  reward_mean_list=np.zeros((iterations,1))
  reward_sd_list=np.zeros((iterations,1))
  cost_mean_list=np.zeros((iterations,1))
  cost_sd_list=np.zeros((iterations,1))

  while i<iterations:
    print('iteration={}' .format(i))
    policy1, policy2, policy3=calculate_soft_policy(omega,theta,lam,gamma,num_action)
    feature, reward_mean, reward_sd, cost, cost_sd=feature_cost_mean_sd(policy1,policy2,policy3,initial_state,num_trials,theta)
    divergence=KL(policy1,policy2,policy3,expert_policy1,expert_policy2,expert_policy3)

    divergence_list[i]=divergence
    reward_mean_list[i]=reward_mean
    reward_sd_list[i]=reward_sd
    cost_mean_list[i]=cost
    cost_sd_list[i]=cost_sd
    feature_gradient=feature-feature_empirical
    cost_gradient=cost-cost_empirical

    print('divergence is {}' .format(divergence))
    print('reward is {}' .format(reward_mean))
    print('cost is {}' .format(cost))
    print('feature_gradient is {}' .format(feature_gradient))
    print('cost_gradient is {}' .format(cost_gradient))

    if i <=80:
      rate=5**(i/20)
    else:  
      rate=100

    omega=omega-(1.0/200000*rate)*feature_gradient
    lam=lam+(1.0/50000)*cost_gradient

    print('omega is {}' .format(omega))
    print('lam is {}' .format(lam))
 
    i=i+1

  divergence_file=open("divergence_file.txt","w")
  for entry in divergence_list:
    np.savetxt(divergence_file,entry)
  divergence_file.close()


  reward_mean_file=open("centralized_reward_mean_file.txt","w")
  for entry in reward_mean_list:
    np.savetxt(reward_mean_file,entry)
  reward_mean_file.close()

  reward_sd_file=open("centralized_reward_sd_file.txt","w")
  for entry in reward_sd_list:
    np.savetxt(reward_sd_file,entry)
  reward_sd_file.close()

  cost_mean_file=open("centralized_cost_mean_file.txt","w")
  for entry in cost_mean_list:
    np.savetxt(cost_mean_file,entry)
  cost_mean_file.close()

  cost_sd_file=open("centralized_cost_sd_file.txt","w")
  for entry in cost_sd_list:
    np.savetxt(cost_sd_file,entry)
  cost_sd_file.close()
  

initial_state=np.mat([0,8,0,4,0,0]).T
num_action=9
gamma=0.9
lam_E=0.2
omega_E=np.mat([1.5,-1.5,1.5,-1.5]).T
theta_E=np.mat([0.0,0.0,0.0,0.0,0.013392000000000001,0.10825200000000006,0.0,0.0,0.15512400000000026,0.24886800000000028,0.32252400000000037,0.02566799999999883,0.09597600000000002,0.03682799999999968,0.07477200000000003,0.0,0.0,0.0,0.0,0.03459599999999999,0.017856,0.3002040000000001,0.0,0.004464,0.06696,0.12164400000000004,0.07923600000000007,0.10602,0.13726800000000006,0.13168800000000008,0.055799999999999975,0.021204,0.004464,0.033479999999999996,0.042408,0.026784000000000002,0.010044,0.010043999999999999,0.00558,0.05914799999999999,0.027899999999999998,0.005580000000000265,0.014508000000000004,0.342612,0.9910080000000009,0.9329760000000007,0.8470440000000007,0.7343280000000004,0.5356800000000003,0.3147120000000001,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.03682799999999999,0.0,0.010043999999999999,0.0,0.020088,0.6338880000000002,0.0,0.0]).T

num_trials=100

a=np.loadtxt("optimal_trajectory_file.txt",dtype=float)
trajectories=a.reshape(30*num_trials,9)

feature_empirical=empirical_feature_counts(trajectories,num_trials)
cost_empirical=empirical_cost_counts(trajectories,num_trials,theta_E)
constraint_empirical=empirical_constraint_counts(trajectories,num_trials,lam_E)

omega=np.mat([0.0,0.0,0.0,0.0,0.0,0.0]).T
lam=0.0

last_interation_inner_loop(initial_state,omega,lam,theta_E,gamma,num_action,num_trials,cost_empirical, feature_empirical)
#inner_loop(initial_state,omega1,omega2,omega3,omega4,lam1,lam2,lam3,lam4,theta_E,gamma,num_action,num_trials,cost_empirical1,cost_empirical2,cost_empirical3,cost_empirical4, feature_empirical1,feature_empirical2,feature_empirical3,feature_empirical4,num_data1,num_data2,num_data3,num_data4)
#theta=np.mat([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).T
#centralized_outer_loop(theta,initial_state,num_trials,constraint_empirical1,constraint_empirical2,constraint_empirical3,constraint_empirical4,num_data1,num_data2,num_data3,num_data4)


#saved_policy=policy.reshape(9*9*9,9*9*9)
#soft_policy_file=open("bellman_policy_file.txt","w")

#for entry in saved_policy:
#  np.savetxt(soft_policy_file,entry)
#soft_policy_file.close()
#distribution=np.loadtxt("bellman_policy_file.txt",dtype=float)
#policy=distribution.reshape(9,9,9,9,num_action,num_action)
#trajectory=trial(initial_state,policy,num_action)
#print(trajectory)




















