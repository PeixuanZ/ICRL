import numpy as np
from math import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, random, time


def draw_env():
  fig=plt.figure()
  gs=gridspec.GridSpec(3,6)
  ax1=fig.add_subplot(gs[:,0:5])
  ax2=fig.add_subplot(gs[0,5])
  ax3=fig.add_subplot(gs[1,5])
  ax4=fig.add_subplot(gs[2,5])
  ax1.axis('scaled')
  ax1.set_xticks(np.linspace(0,9,10))
  ax1.set_yticks(np.linspace(0,9,10))
  ax1.axis([0,9,0,9])
  ax1.grid(linestyle='-',color='black')
  obstacle1=plt.Rectangle((4,0),1,1,linewidth=2,edgecolor='black',facecolor='none')
  obstacle2=plt.Rectangle((4,1),1,1,linewidth=2,edgecolor='black',facecolor='none')
  obstacle3=plt.Rectangle((4,2),1,1,linewidth=2,edgecolor='black',facecolor='none')
  obstacle4=plt.Rectangle((4,6),1,1,linewidth=2,edgecolor='black',facecolor='none')
  obstacle5=plt.Rectangle((4,7),1,1,linewidth=2,edgecolor='black',facecolor='none')
  obstacle6=plt.Rectangle((4,8),1,1,linewidth=2,edgecolor='black',facecolor='none')
  obstacle7=plt.Rectangle((4,3),1,1,linewidth=2,edgecolor='black',facecolor='none')
  obstacle8=plt.Rectangle((3,4),1,1,linewidth=2,edgecolor='black',facecolor='none')
  obstacle9=plt.Rectangle((2,5),1,1,linewidth=2,edgecolor='black',facecolor='none')
  obstacle10=plt.Rectangle((5,4),1,1,linewidth=2,edgecolor='black',facecolor='none')
  obstacle11=plt.Rectangle((6,5),1,1,linewidth=2,edgecolor='black',facecolor='none')

  ax1.add_patch(obstacle1)
  ax1.add_patch(obstacle2)
  ax1.add_patch(obstacle3)
  ax1.add_patch(obstacle4)
  ax1.add_patch(obstacle5)
  ax1.add_patch(obstacle6)
  ax1.add_patch(obstacle7)
  ax1.add_patch(obstacle8)
  ax1.add_patch(obstacle9)
  ax1.add_patch(obstacle10)
  ax1.add_patch(obstacle11)
  #ax1.add_patch(plt.Rectangle((1,6),1,1,linewidth=2,edgecolor='black',facecolor='none'))
  ax1.scatter(4.5,0.5,s=160,c="r",marker="x")
  ax1.scatter(4.5,1.5,s=160,c="r",marker="x")
  ax1.scatter(4.5,2.5,s=160,c="r",marker="x")
  ax1.scatter(4.5,6.5,s=160,c="r",marker="x")
  ax1.scatter(4.5,7.5,s=160,c="r",marker="x")
  ax1.scatter(4.5,8.5,s=160,c="r",marker="x")
  ax1.scatter(4.5,3.5,s=160,c="r",marker="x")
  ax1.scatter(3.5,4.5,s=160,c="r",marker="x")
  ax1.scatter(2.5,5.5,s=160,c="r",marker="x")
  ax1.scatter(5.5,4.5,s=160,c="r",marker="x")
  ax1.scatter(6.5,5.5,s=160,c="r",marker="x")
  ax1.text(0.35,0.35,'$s_0^{\prime\prime}$',fontsize=10)
  ax1.text(8.35,0.35,'$s_G^{\prime\prime}$',fontsize=10)
  ax1.text(0.35,4.35,'$s_0^{\prime}$',fontsize=10)
  ax1.text(8.35,4.35,'$s_G^{\prime}$',fontsize=10)
  ax1.text(0.35,8.35,'$s_0$',fontsize=10)
  ax1.text(8.35,8.35,'$s_G$',fontsize=10)
  ax1.set_title('States')
  for axi in (ax1.xaxis, ax1.yaxis):
      for tic in axi.get_major_ticks():
          tic.tick1On = tic.tick2On = False
          #tic.label1On = tic.label2On = False

  ax2.axis('scaled')
  ax2.set_xticks(np.linspace(0,3,4))
  ax2.set_yticks(np.linspace(0,3,4))
  ax2.axis([0,3,0,3])
  ax2.grid(linestyle='-',color='black')
  ax2.set_title('Actions')
  ax2.set_xlabel('Expert 1')
  obstacle12=plt.Rectangle((0,0),1,1,linewidth=1.5,edgecolor='black',facecolor='none')
  obstacle13=plt.Rectangle((2,0),1,1,linewidth=1.5,edgecolor='black',facecolor='none')
  ax2.add_patch(obstacle12)
  ax2.add_patch(obstacle13)
  ax2.scatter(0.5,0.5,s=80,c="r",marker="x")
  ax2.scatter(2.5,0.5,s=80,c="r",marker="x")
  ax2.arrow(1.5,1.5,0,1,head_width=0.2, head_length=0.2,color='silver')
  ax2.arrow(1.5,1.5,0.707,0.707,head_width=0.2, head_length=0.2,color='silver')
  ax2.arrow(1.5,1.5,1,0,head_width=0.2, head_length=0.2,color='silver')
  ax2.arrow(1.5,1.5,0.707,-0.707,head_width=0.2, head_length=0.2,color='silver')
  ax2.arrow(1.5,1.5,0,-1,head_width=0.2, head_length=0.2,color='silver')
  ax2.arrow(1.5,1.5,-0.707,-0.707,head_width=0.2, head_length=0.2,color='silver')
  ax2.arrow(1.5,1.5,-1,0,head_width=0.2, head_length=0.2,color='silver')
  ax2.arrow(1.5,1.5,-0.707,0.707,head_width=0.2, head_length=0.2,color='silver')
  for axi in (ax2.xaxis, ax2.yaxis):
      for tic in axi.get_major_ticks():
          tic.tick1On = tic.tick2On = False
          tic.label1On = tic.label2On = False

  ax3.axis('scaled')
  ax3.set_xticks(np.linspace(0,3,4))
  ax3.set_yticks(np.linspace(0,3,4))
  ax3.axis([0,3,0,3])
  ax3.grid(linestyle='-',color='black')
  #ax3.set_title('Actions')
  ax3.set_xlabel('Expert 2')
  obstacle16=plt.Rectangle((0,2),1,1,linewidth=1.5,edgecolor='black',facecolor='none')
  obstacle17=plt.Rectangle((2,0),1,1,linewidth=1.5,edgecolor='black',facecolor='none')
  ax3.add_patch(obstacle16)
  ax3.add_patch(obstacle17)
  ax3.scatter(2.5,0.5,s=80,c="r",marker="x")
  ax3.scatter(0.5,2.5,s=80,c="r",marker="x")
  ax3.arrow(1.5,1.5,0,1,head_width=0.2, head_length=0.2,color='silver')
  ax3.arrow(1.5,1.5,0.707,0.707,head_width=0.2, head_length=0.2,color='silver')
  ax3.arrow(1.5,1.5,1,0,head_width=0.2, head_length=0.2,color='silver')
  ax3.arrow(1.5,1.5,0.707,-0.707,head_width=0.2, head_length=0.2,color='silver')
  ax3.arrow(1.5,1.5,0,-1,head_width=0.2, head_length=0.2,color='silver')
  ax3.arrow(1.5,1.5,-0.707,-0.707,head_width=0.2, head_length=0.2,color='silver')
  ax3.arrow(1.5,1.5,-1,0,head_width=0.2, head_length=0.2,color='silver')
  ax3.arrow(1.5,1.5,-0.707,0.707,head_width=0.2, head_length=0.2,color='silver')
  for axi in (ax3.xaxis, ax3.yaxis):
      for tic in axi.get_major_ticks():
          tic.tick1On = tic.tick2On = False
          tic.label1On = tic.label2On = False

  ax4.axis('scaled')
  ax4.set_xticks(np.linspace(0,3,4))
  ax4.set_yticks(np.linspace(0,3,4))
  ax4.axis([0,3,0,3])
  ax4.grid(linestyle='-',color='black')
  #ax4.set_title('Actions')
  ax4.set_xlabel('Expert 3')
  obstacle14=plt.Rectangle((0,2),1,1,linewidth=1.5,edgecolor='black',facecolor='none')
  obstacle15=plt.Rectangle((2,2),1,1,linewidth=1.5,edgecolor='black',facecolor='none')
  ax4.add_patch(obstacle14)
  ax4.add_patch(obstacle15)
  ax4.scatter(0.5,2.5,s=80,c="r",marker="x")
  ax4.scatter(2.5,2.5,s=80,c="r",marker="x")
  ax4.arrow(1.5,1.5,0,1,head_width=0.2, head_length=0.2,color='silver')
  ax4.arrow(1.5,1.5,0.707,0.707,head_width=0.2, head_length=0.2,color='silver')
  ax4.arrow(1.5,1.5,1,0,head_width=0.2, head_length=0.2,color='silver')
  ax4.arrow(1.5,1.5,0.707,-0.707,head_width=0.2, head_length=0.2,color='silver')
  ax4.arrow(1.5,1.5,0,-1,head_width=0.2, head_length=0.2,color='silver')
  ax4.arrow(1.5,1.5,-0.707,-0.707,head_width=0.2, head_length=0.2,color='silver')
  ax4.arrow(1.5,1.5,-1,0,head_width=0.2, head_length=0.2,color='silver')
  ax4.arrow(1.5,1.5,-0.707,0.707,head_width=0.2, head_length=0.2,color='silver')
  for axi in (ax4.xaxis, ax4.yaxis):
      for tic in axi.get_major_ticks():
          tic.tick1On = tic.tick2On = False
          tic.label1On = tic.label2On = False
  plt.savefig('environment.pdf')  
  plt.show()

def single_expert_stochastic_dynamics(state,action):
  action=action.item()
  x=np.copy(state[0].item())
  y=np.copy(state[1].item())
  sign=np.random.uniform()
  if sign<=0.2:
    the_next_state=[x,y]
  else:
    if action==0:
      the_next_state=[x,y]
    elif action==1:
      if y==8:
        the_next_state=[x,y]
      else:
        the_next_state=[x,y+1]
    elif action==2:
      if y==0:
        the_next_state=[x,y]
      else:
        the_next_state=[x,y-1]
    elif action==3:
      if x==0:
        the_next_state=[x,y]
      else:
        the_next_state=[x-1,y]
    elif action==4:
      if x==8:
        the_next_state=[x,y]
      else:
        the_next_state=[x+1,y]
    elif action==5:
      if x==0 or y==8:
        the_next_state=[x,y]
      else:
        the_next_state=[x-1,y+1]
    elif action==6:
      if x==8 or y==8:
        the_next_state=[x,y]
      else:
        the_next_state=[x+1,y+1]
    elif action==7:
      if x==0 or y==0:
        the_next_state=[x,y]
      else:
        the_next_state=[x-1,y-1]
    elif action==8:
      if x==8 or y==0:
        the_next_state=[x,y]
      else:
        the_next_state=[x+1,y-1]
  return np.mat(the_next_state).T

def single_expert_dynamics(state,action):
  x=np.copy(state[0].item())
  y=np.copy(state[1].item())
  if action==0:
    the_next_state=[x,y]
  elif action==1:
    if y==8:
      the_next_state=[x,y]
    else:
      the_next_state=[x,y+1]
  elif action==2:
    if y==0:
      the_next_state=[x,y]
    else:
      the_next_state=[x,y-1]
  elif action==3:
    if x==0:
      the_next_state=[x,y]
    else:
      the_next_state=[x-1,y]
  elif action==4:
    if x==8:
      the_next_state=[x,y]
    else:
      the_next_state=[x+1,y]
  elif action==5:
    if x==0 or y==8:
      the_next_state=[x,y]
    else:
      the_next_state=[x-1,y+1]
  elif action==6:
    if x==8 or y==8:
      the_next_state=[x,y]
    else:
      the_next_state=[x+1,y+1]
  elif action==7:
    if x==0 or y==0:
      the_next_state=[x,y]
    else:
      the_next_state=[x-1,y-1]
  elif action==8:
    if x==8 or y==0:
      the_next_state=[x,y]
    else:
      the_next_state=[x+1,y-1]
  return np.mat(the_next_state).T

def single_expert_deterministic_dynamics(state,action):
  action=action.item()
  x=np.copy(state[0].item())
  y=np.copy(state[1].item())
  if action==0:
    the_next_state=[x,y]
  elif action==1:
    the_next_state=[x,y+1]
  elif action==2:
    the_next_state=[x,y-1]
  elif action==3:
    the_next_state=[x-1,y]
  elif action==4:
    the_next_state=[x+1,y]
  elif action==5:
    the_next_state=[x-1,y+1]
  elif action==6:
    the_next_state=[x+1,y+1]
  elif action==7:
    the_next_state=[x-1,y-1]
  elif action==8:
    the_next_state=[x+1,y-1]
  return np.mat(the_next_state).T


def feature1(state,action):
  next_state=single_expert_deterministic_dynamics(state,action)
  if next_state.item(0)<0 or next_state.item(0)>8 or next_state.item(1)<0 or next_state.item(1)>8:
    return np.mat([0.0,8.0]).T
  elif state.item(0)==8 and state.item(1)==8:
    return np.mat([40.0,0.0]).T
  else:
    #return np.mat([0.0,0.0]).T
    return 1.0*np.mat([np.copy(state.item(0)),(8-np.copy(state.item(1)))]).T

def feature2(state,action):
  next_state=single_expert_deterministic_dynamics(state,action)
  if next_state.item(0)<0 or next_state.item(0)>8 or next_state.item(1)<0 or next_state.item(1)>8:
    return np.mat([0.0,8.0]).T
  elif state.item(0)==8 and state.item(1)==4:
    return np.mat([40.0,0.0]).T
  else:
    #return np.mat([0.0,0.0]).T
    return 1.0*np.mat([np.copy(state.item(0)),abs(4-np.copy(state.item(1)))]).T

def feature3(state,action):
  next_state=single_expert_deterministic_dynamics(state,action)
  if next_state.item(0)<0 or next_state.item(0)>8 or next_state.item(1)<0 or next_state.item(1)>8:
    return np.mat([0.0,8.0]).T
  elif state.item(0)==8 and state.item(1)==0:
    return np.mat([40.0,0.0]).T
  else:
    #return np.mat([0.0,0.0]).T
    return 1.0*state

def expert1_reward(omega,state,action):
  reward=np.dot(omega.T,feature1(state,action))
  return reward.item()

def expert2_reward(omega,state,action):
  reward=np.dot(omega.T,feature2(state,action))
  return reward.item()

def expert3_reward(omega,state,action):
  reward=np.dot(omega.T,feature3(state,action))
  return reward.item()

def single_expert_basis_state_constraint(single_expert_state):
  x=np.copy(single_expert_state.item(0))
  y=np.copy(single_expert_state.item(1))
  obstacle=np.mat(np.zeros((20,1)))
  if x==4 and y==0:
    obstacle[0]=1000.0
  if x==4 and y==1:
    obstacle[1]=1000.0
  if x==4 and y==2:
    obstacle[2]=1000.0
  if x==4 and y==3:
    obstacle[3]=1000.0
  if x==3 and y==4:
    obstacle[4]=1000.0
  if x==2 and y==5:
    obstacle[5]=1000.0 
  if x==5 and y==4:
    obstacle[6]=1000.0
  if x==6 and y==5:
    obstacle[7]=1000.0
  if x==4 and y==6:
    obstacle[8]=1000.0
  if x==4 and y==7:
    obstacle[9]=1000.0
  if x==4 and y==8:
    obstacle[10]=1000.0
  if x==3 and y==6:
    obstacle[11]=1000.0
  if x==2 and y==6:
    obstacle[12]=1000.0
  if x==3 and y==7:
    obstacle[13]=1000.0
  if x==2 and y==7:
    obstacle[14]=1000.0
  if x==1 and y==2:
    obstacle[15]=1000.0
  if x==0 and y==2:
    obstacle[16]=1000.0
  if x==1 and y==3:
    obstacle[17]=1000.0
  if x==0 and y==3:
    obstacle[18]=1000.0
  if x==1 and y==4:
    obstacle[19]=1000.0
  return obstacle

def expert_1_basis_constraint(state,action):
  action=action.item()
  state_basis_constraint=single_expert_basis_state_constraint(state)
  action_constraint=np.mat(np.zeros((2,1)))
  if action==7:
    action_constraint[0]=1000.0
  if action==8:
    action_constraint[1]=1000.0
  return np.vstack((state_basis_constraint,action_constraint))     #dimension is 22

def expert_2_basis_constraint(state,action):
  action=action.item()
  state_basis_constraint=single_expert_basis_state_constraint(state)
  action_constraint=np.mat(np.zeros((2,1)))
  if action==5:
    action_constraint[0]=1000.0
  if action==8:
    action_constraint[1]=1000.0
  return np.vstack((state_basis_constraint,action_constraint))     #dimension is 22

def expert_3_basis_constraint(state,action):
  action=action.item()
  state_basis_constraint=single_expert_basis_state_constraint(state)
  action_constraint=np.mat(np.zeros((4,1)))
  if action==5:
    action_constraint[0]=1000.0
  if action==6:
    action_constraint[1]=1000.0
  if action==7:
    action_constraint[2]=1000.0
  if action==8:
    action_constraint[3]=1000.0
  return np.vstack((state_basis_constraint,action_constraint))     #dimension is 24

def expert1_cost(theta,state,action):
  constraint_vector=expert_1_basis_constraint(state,action)
  cost=np.dot(theta.T,constraint_vector) 
  return cost.item()    

def expert2_cost(theta,state,action):
  constraint_vector=expert_2_basis_constraint(state,action)
  cost=np.dot(theta.T,constraint_vector) 
  return cost.item()    

def expert3_cost(theta,state,action):
  constraint_vector=expert_3_basis_constraint(state,action)
  cost=np.dot(theta.T,constraint_vector)
  return cost.item()    

#draw_env()
















