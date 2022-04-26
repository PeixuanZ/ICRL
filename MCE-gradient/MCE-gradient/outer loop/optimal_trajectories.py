import numpy as np
from env import single_expert_stochastic_dynamics

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


initial_state=np.mat([0,8,0,4,0,0]).T
num_action=9

distribution1=np.loadtxt("optimal_policy1_file.txt",dtype=float)
policy1=distribution1.reshape(9,9,num_action)
distribution2=np.loadtxt("optimal_policy2_file.txt",dtype=float)
policy2=distribution2.reshape(9,9,num_action)
distribution3=np.loadtxt("optimal_policy3_file.txt",dtype=float)
policy3=distribution3.reshape(9,9,num_action)

trajectory_file=open("optimal_trajectory_file.txt","w")

num_trials=100
for i in range(num_trials):
  trajectory=np.copy(optimal_trial(initial_state,policy1,policy2,policy3,num_action))
  for entry in trajectory:
    np.savetxt(trajectory_file,entry)
trajectory_file.close()
#a=np.loadtxt("optimal_trajectory_file.txt",dtype=float)
#print(a.reshape(30,6))

## the data type: we have 100 trajectories, each trajectory has 31 states and each state has 4 components. The trajectory file has 4x31x100 lines.
