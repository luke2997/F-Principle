import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
# %matplotlib inline


def func0(x):
# Pick (x,y) from this function 
    y_sin = np.sin(x) + np.sin(4 * x) + np.sin(6 * x)
    return y_sin
  
  
x_start = -10 ### start point of input
x_end = 10 # end point of input

test_size = int(201)  ### test size
train_size = int(61);  ### training size

# initialization for variables
x_test = np.reshape(np.linspace(x_start, x_end, num=test_size,
                                                  endpoint=True),[test_size,1])

x_train = np.reshape(np.linspace(x_start, x_end, num=train_size,
                                                  endpoint=True),[train_size,1])

y_test = func0(x_test)
y_train = func0(x_train)
y_test = np.round(y_test, decimals=3)
y_train = np.round(y_train, decimals=3)

position = np.concatenate((x_train, y_train), axis=1)

print(len(position)) #61

import random

def add_and_remove_edges(G, p_new_connection, p_remove_connection):    
    '''    
    for each node,    
      add a new connection to random other node, with prob p_new_connection,    
      remove a connection, with prob p_remove_connection    

    operates on G in-place    
    '''                
    new_edges = []    
    rem_edges = []    

    for node in G.nodes():    
        # find the other nodes this one is connected to    
        connected = [to for (fr, to) in G.edges(node)]    
        # and find the remainder of nodes, which are candidates for new edges   
        unconnected = [n for n in G.nodes() if not n in connected]    
        
        # print(len(G.nodes()))

        # probabilistically add a random edge    
        if len(unconnected): # only try if new edge is possible    
            if random.random() < p_new_connection:    
                new = random.choice(unconnected)    
                G.add_edge(node, new)       
                new_edges.append( (node, new) )    
                # book-keeping, in case both add and remove done in same cycle  
                unconnected.remove(new)    
                connected.append(new)    

        # probabilistically remove a random edge    
        if len(connected): # only try if an edge exists to remove    
            if random.random() < p_remove_connection:    
                remove = random.choice(connected)    
                G.remove_edge(node, remove)        
                rem_edges.append( (node, remove) )    
                # book-keeping, in case lists are important later?    
                connected.remove(remove)    
                unconnected.append(remove)    
    return rem_edges, new_edges 
  
  
p_new_connection = 0.1
p_remove_connection = 0.1

G = nx.karate_club_graph()

plt.figure(1); plt.clf()
fig, ax = plt.subplots(2,1, num=1, sharex=True, sharey=True)
pos = position
nx.draw_networkx(G, pos=pos, ax=ax[0])

rem_edges, new_edges = add_and_remove_edges(G, p_new_connection, p_remove_connection)

nx.draw_networkx(G, pos=pos, ax=ax[1])
nx.draw_networkx_edges(G, pos=pos, ax=ax[1], edgelist=new_edges,
                       edge_color='b', width=4)
                       
G.add_edges_from(rem_edges)
nx.draw_networkx_edges(G, pos=pos, ax=ax[1], edgelist=rem_edges,
                       edge_color='r', style='dashed', width=4)
G.remove_edges_from(rem_edges)

plt.show() 
