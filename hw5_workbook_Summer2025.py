# Databricks notebook source
# MAGIC %md
# MAGIC ## HW 5 - Page Rank
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Summer 2025`__
# MAGIC
# MAGIC > Updated: 07/08/2025
# MAGIC
# MAGIC This is also a Team Homework, using the same team as the Final Project. **What's the catch? There are 4 questions, each student in the team has to author at least one question. Question 4 is the hardest, so for teams of 5 or more, this question can be shared**
# MAGIC
# MAGIC **Clone this notebook to be able to run it!**
# MAGIC
# MAGIC
# MAGIC In Weeks 9 and 10 you discussed key concepts related to graph based algorithms and implemented SSSP. In this final homework assignment you'll implement distributed PageRank using some data from Wikipedia. This homework it's a team homework, thus every team will only submit one notebook. 
# MAGIC
# MAGIC By the end of this homework you should be able to:  
# MAGIC * ... __compare/contrast__ adjacency matrices and lists as representations of graphs for parallel computation.
# MAGIC * ... __explain__ the goal of the PageRank algorithm using the concept of an infinite Random Walk.
# MAGIC * ... __define__ a Markov chain including the conditions underwhich it will converge.
# MAGIC * ... __identify__ what modifications must be made to the web graph inorder to leverage Markov Chains.
# MAGIC * ... __implement__ distributed PageRank in Spark.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notebook Set-Up
# MAGIC Before starting your homework run the following cells to confirm your setup.

# COMMAND ----------

pip install graphframes networkx

# COMMAND ----------

# RUN CELL AS IS - Imports
import re
import ast
import time
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from graphframes import *
from pyspark.sql import functions as F

# COMMAND ----------

# RUN CELL AS IS - Spark Context
sc = spark.sparkContext
spark

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Check

# COMMAND ----------

# RUN THIS CELL AS IS. You should see all-pages-indexed-in.txt, all-pages-indexed-out.txt and indices.txt in the results. If you do not see these, please let an Instructor or TA know.
DATA_PATH = "dbfs:/mnt/mids-w261/HW5/HW5/"
display(dbutils.fs.ls(DATA_PATH))

# COMMAND ----------

# MAGIC %md
# MAGIC # Question 1: Graph Processing Theory
# MAGIC
# MAGIC # Author: Trevor Lang
# MAGIC
# MAGIC ## a. Distribute Graph Processing
# MAGIC Chapter 5 from Lin & Dyer gave you a high level introduction to graph algorithms and concernts that come up when trying to perform distributed computations over them. The questions below are designed to make sure you captured the key points from this reading. 
# MAGIC
# MAGIC ### Q1.a Tasks:
# MAGIC * __a) Multiple Choice:__ Other than their size/scale, what makes graphs uniquely challenging to work with in the map-reduce paradigm? *(__HINT__: Don't think in terms of any specific algorithm. Think in terms of the nature of the graph data structure itself).*
# MAGIC
# MAGIC ## b. Representing Graphs
# MAGIC In class you saw examples of adjacency matrix and adjacency list representations of graphs. These data structures were probably familiar from HW3, though we hadn't before talked about them in the context of graphs. In this question we'll discuss some of the tradeoffs associated with these representations. __`NOTE:`__ We'll use the graph from Figure 5.1 in Lin & Dyer as a toy example. For convenience in the code below we'll label the nodes `A`, `B`, `C`, `D`, and `E` instead of \\(n{_1}\\), \\(n{_2}\\), etc but otherwise you should be able to follow along & check our answers against those in the text.
# MAGIC
# MAGIC
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/HW5/Lin-Dyer-graph-Q1.png?raw=true" width=50%>
# MAGIC
# MAGIC ### Q1.b Tasks:
# MAGIC
# MAGIC * __a) Fill in the blanks:__ Relatively speaking, is the graph you described in Figure 5.1 in Lin & Dyer "sparse" or "dense"?  Explain how sparsity/density impacts the adjacency matrix and adjacency list representations of a graph. 
# MAGIC * __b) Fill in the blanks:__ Run the provided code to create and plot our toy graph. Is this graph directed or undirected? Explain how the adjacency matrices for directed graphs will differ from those of undirected graphs.
# MAGIC * __c) Code:__ Fill in the missing code to complete the function `get_adj_matr()`.
# MAGIC * __d) Code:__ Fill in the missing code to complete the function `get_adj_list()`.
# MAGIC * __e) Multiple Choice:__ Which is the correct edge list for node `B`?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q1.a Student Answers:
# MAGIC > __a)__ Highlight the correct answer (Add ** before and after your selected choice):
# MAGIC * **It can be hard to represent graphs as distinct records that can be processed separately from each other.**
# MAGIC * Graphs contain nodes and edges, which cannot easily be converted into key-value pairs.
# MAGIC * You need to store the entire graph in memory, so we can only work with small datasets.
# MAGIC
# MAGIC ### Q1.b Student Answers
# MAGIC > __a)__ Fill in the blanks with any of these possible statements based on the following category.
# MAGIC   > * type = [**__sparse__**, dense]
# MAGIC   > * number = [**__N^2 edges or 5^2 = 25__**, N(N-1) edges or 5*4 = 20]
# MAGIC   > * efficiency = [**__more efficient__**, less efficient]
# MAGIC
# MAGIC * This is a relatively **sparse** matrix. If self loops are allowed, then we could have a total of **N^2 edges or 5^2 = 25** edges. Otherwise, if self loops are not allowed, we could have **N(N-1) edges or 5*4 = 20** edges. Out of a total possible edges, this graph only has 9 edges. For sparse graphs, their adjacency list representation will be **more efficient** than their adjacency matrix (memory wise). 
# MAGIC
# MAGIC
# MAGIC > __b)__ Fill in the blanks with any of these possible statements based on the following category.
# MAGIC   > * graph_type = [**__directed__**, undirected]
# MAGIC   > * symmetric = [**__asymmetric__**, **__symmetric__**]
# MAGIC
# MAGIC * This is a **directed** graph. Because we see the edge ('A', 'B') in the graph, but there is no corresponding ('B', 'A'). For directed graphs, the adjacency matrix will be **asymmetric** Whereas for undirected graphs, it will be **symmetric**. 
# MAGIC
# MAGIC > __e)__ Highlight the correct answer (Add ** before and after your selected choice):
# MAGIC * **['C','E']**
# MAGIC * ['D']
# MAGIC * ['A', 'B', 'C']
# MAGIC * ['B', 'D']

# COMMAND ----------

# part 1b.a - a graph is just a list of nodes and edges (RUN THIS CELL AS IS)
TOY_GRAPH = {'nodes':['A', 'B', 'C', 'D', 'E'],
             'edges':[('A', 'B'), ('A', 'D'), ('B', 'C'), ('B', 'E'), ('C', 'D'), 
                      ('D', 'E'), ('E', 'A'),('E', 'B'), ('E', 'C')]}

# COMMAND ----------

# part 1b.a - simple visualization of our toy graph using nx (RUN THIS CELL AS IS)
G = nx.DiGraph()
G.add_nodes_from(TOY_GRAPH['nodes'])
G.add_edges_from(TOY_GRAPH['edges'])
display(nx.draw_networkx(G, pos=nx.circular_layout(G), with_labels=True, alpha = 0.5))

# COMMAND ----------

# part 1b.c - adjacency matrix function
def get_adj_matr(graph):
    """
    Function to create an adjacency matrix representation of a graph.
    arg:
        graph - (dict) of 'nodes' : [], 'edges' : []
    returns:
        pd.DataFrame with entry i,j representing an edge from node i to node j
    """
    n = len(graph['nodes'])
    adj_matr = pd.DataFrame(0, columns = graph['nodes'], index = graph['nodes'])
    ############### YOUR CODE HERE ##################
    for src, dst in graph['edges']:
        adj_matr.loc[src, dst] = 1
    ############### (END) YOUR CODE #################
    return adj_matr

# COMMAND ----------

# part 1b.c - take a look (RUN THIS CELL AS IS)
TOY_ADJ_MATR = get_adj_matr(TOY_GRAPH)
print(TOY_ADJ_MATR)

# COMMAND ----------

# part 1b.d - adjacency list function
def get_adj_list(graph):
    """
    Function to create an adjacency list representation of a graph.
    arg:
        graph - (dict) of 'nodes' : [], 'edges' : []
    returns:
        dictionary of the form {node : [list of edges]}
    """
    adj_list = {node: [] for node in graph['nodes']}
    ############### YOUR CODE HERE ##################
    for src, dst in graph['edges']:
        adj_list[src].append(dst)
    ############### (END) YOUR CODE #################
    return adj_list

# COMMAND ----------

# part 1b.d - take a look (RUN THIS CELL AS IS)
TOY_ADJ_LIST = get_adj_list(TOY_GRAPH)
print(TOY_ADJ_LIST)

# COMMAND ----------

# MAGIC %md
# MAGIC # Question 2: Markov Chains, Random Walks and PageRank
# MAGIC # Author: Omar Zu'bi (2A) 
# MAGIC
# MAGIC ## a. Markov Chains and Random Walks
# MAGIC As you know from your readings and in class discussions, the PageRank algorithm takes advantage of the machinery of Markov Chains to compute the relative importance of a webpage using the hyperlink structure of the web (we'll refer to this as the 'web-graph'). A Markov Chain is a discrete-time stochastic process. The stochastic matrix has a principal left eigen vector corresponding to its largest eigen value which is one. A Markov chain's probability distribution over its states may be viewed as a probability vector. This steady state probability for a state is the PageRank of the corresponding webpage. In this question we'll briefly discuss a few concepts that are key to understanding the math behind PageRank. 
# MAGIC
# MAGIC General note: We use a sped-up convergence method for the `power_iteration()` function. You should speed up convergence by multiplying by the tMatrix twice. `tMatrix = tMatrix.dot(tMatrix)`. So instead of:
# MAGIC
# MAGIC ```
# MAGIC Step 0: xInit * tMatrix
# MAGIC Step 1: xInit * tMatrix^2
# MAGIC Step 2: xInit * tMatrix^3
# MAGIC Step 3: xInit * tMatrix^4
# MAGIC ...
# MAGIC ```
# MAGIC
# MAGIC you should instead consider
# MAGIC
# MAGIC ```
# MAGIC Step 0: xInit * tMatrix
# MAGIC Step 1: xInit * tMatrix^2
# MAGIC Step 2: xInit * tMatrix^4
# MAGIC Step 3: xInit * tMatrix^8
# MAGIC ...
# MAGIC ```
# MAGIC
# MAGIC This may vary slightly from what you see in Demo 10
# MAGIC
# MAGIC ### Q2.a Tasks:
# MAGIC
# MAGIC * __a) Multiple Choice:__ It is common to explain PageRank using the analogy of a web surfer who visits a page, randomly clicks a link on that page, and repeats ad infinitum. In the context of this hypothetical infinite random walk across web pages on the internet, which of the following choices most clearly describes the event that the __teleportation__ represents?
# MAGIC
# MAGIC * __b) Fill in the blanks:__ What is the "Markov Property" and what does it mean in the context of PageRank?
# MAGIC
# MAGIC * __c) Fill in the blanks:__ A Markov chain consists of \\(n\\) states plus an \\(n\times n\\) transition probability matrix. In the context of PageRank & a random walk over the WebGraph, what are the \\(n\\) states? what implications does this have about the size of the transition matrix?
# MAGIC
# MAGIC * __d) Code + Numerical Answer:__ Fill in the code below to compute the transition matrix for the toy graph from question 1. What is the value in the middle column of the last row (the probability of transitioning from node `E` to `C`)? Include the leading digit in front of the decimal, and round the number to at least 4 decimal places. Examples: 1.0000 or 0.1234. [__`HINT:`__ _It should be right stochastic. Using numpy this calculation can be done in one line of code._]
# MAGIC
# MAGIC * __e) Code:__ To compute the stable state distribution (i.e. PageRank) of a "nice" graph we can apply the power iteration method - repeatedly multiplying the transition matrix by itself, until the values no longer change. Apply this strategy to your transition matrix from `part d` to find the PageRank for each of the pages in your toy graph. Your code should print the results of each iteration. 
# MAGIC     * __`NOTE 1:`__ _this is a naive approach, we'll unpack what it means to be "nice" in the next question_.
# MAGIC     * __`NOTE 2:`__ _no need to implement a stopping criteria, visual inspection should suffice_.
# MAGIC     * __`NOTE 3:`__ _refer to the General note section above, and use the `tMatrix = tMatrix.dot(tMatrix)` approach for faster convergence_.
# MAGIC
# MAGIC * __f) Numerical Answer:__ How many iterations does it take to converge? Define convergence as the step where you can identify conevergence. Example: If step 4 the value is 0.435 and in step 5 the value is also 0.435, you identified convergence in step 5.
# MAGIC
# MAGIC * __g) Multiple Choice:__ Which node is the least 'central' (i.e., it has the lowest ranked)?
# MAGIC
# MAGIC ## b. Page Rank Theory
# MAGIC Seems easy right? Unfortunately applying this power iteration method directly to the web-graph actually runs into a few problems. In this question we'll tease apart what we meant by a 'nice graph' from before and highlight key modifications we'll have to make to the web-graph when performing PageRank. To start, we'll look at what goes wrong when we try to repeat our strategy from before on a 'not nice' graph.
# MAGIC
# MAGIC __`Additional References:`__ http://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture3/lecture3.html
# MAGIC
# MAGIC ### Q2.b Tasks:
# MAGIC #Author: Elana Wadhwani
# MAGIC * __a) Code + Multiple Choice:__ Run the provided code to create and plot our 'not nice' graph. Fill in the missing code to compute its transition matrix & run the power iteration method from before. What are the first values in steps 1, 2, and 3 of the power iteration method? [__`HINT:`__ _We start the iteration at step number 0. If you start your iteration at step number 1, then you should answer with the values from step 2, 3, and 4 instead_]
# MAGIC
# MAGIC * __b) Multiple Choice:__ What is wrong with what you see in part a? [__`HINT:`__ _there is a visible underlying reason that it isn't converging... try adding up the probabilities in the state vector after each iteration._]
# MAGIC
# MAGIC * __c) Multiple Choice:__  Identify the dangling node in this 'not nice' graph and explain how this node causes the problem in 'a'. How could we modify the transition matrix after each iteration to prevent this problem?
# MAGIC
# MAGIC * __d) Multiple Choice:__ What does it mean for a graph to be irreducible? Is the webgraph naturally irreducible? Choose your reasoning.
# MAGIC
# MAGIC * __e) Multiple Choice:__ What does it mean for a graph to be aperiodic? Is the webgraph naturally aperiodic? Choose your reasoning.
# MAGIC
# MAGIC * __f) Multiple Choices:__ What modifications to the webgraph does PageRank make in order to guarantee aperiodicity and irreducibility? [__`HINT:`__ _select 2 answers_]
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q2.a Student Answers:
# MAGIC > __a)__ Highlight the correct answer (Add ** before and after your selected choice):
# MAGIC * Randomly traveling to a linked site from the current page.
# MAGIC * Deterministically going to a linked site from a the current page.
# MAGIC * **Randomly traveling to a non-linked site from the current page.**
# MAGIC * A self-loop in which the user clicks a link to return the same page.
# MAGIC
# MAGIC > __b)__ Fill in the blanks with any of these possible statements based on the following category.
# MAGIC   > * property = [**memorylessness**, randomness]
# MAGIC   > * assumption = [**the probability of transitioning from one page to another is stable regardless of past browsing history**, the probability of transitioning from one page to another changes according to the past browsing history]
# MAGIC
# MAGIC * The Markov Property is the property of [**memorylessness**]. In the context of Page Rank, this is the assumption that [**the probability of transitioning from one page to another is stable regardless of past browsing history**].
# MAGIC
# MAGIC > __c)__ Fill in the blanks with any of these possible statements based on the following category.
# MAGIC   > * states = [links, **webpages**]
# MAGIC   > * size = [**large**, relatively small]
# MAGIC
# MAGIC * The n states are the [**webpages**]. Therefore, the size of the transition matrix will be [**large**].
# MAGIC
# MAGIC > __d)__ Numeric Answer: 0.3333
# MAGIC
# MAGIC > __f)__ Numeric Answer: 6
# MAGIC
# MAGIC > __g)__ Highlight the correct answer (Add ** before and after your selected choice):
# MAGIC * **A**
# MAGIC * B
# MAGIC * C
# MAGIC * D
# MAGIC * E
# MAGIC

# COMMAND ----------

# Define the toy graph from Q1
nodes = ['A', 'B', 'C', 'D', 'E']
edges = [
    ('A', 'B'), ('A', 'D'), ('B', 'C'), ('B', 'E'),
    ('C', 'D'), ('D', 'E'), ('E', 'A'), ('E', 'B'), ('E', 'C')
]

n = len(nodes)
node_idx = {node: i for i, node in enumerate(nodes)}

### PART D) Get the probability of transitioning from node E to C ###
# Create adjacency matrix
adj_matrix = np.zeros((n, n))
for src, dst in edges:
    adj_matrix[node_idx[src], node_idx[dst]] = 1

# Create transition matrix (right-stochastic)
transition_matrix = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)

# Probability of transitioning from E to C
prob_E_to_C = transition_matrix[node_idx['E'], node_idx['C']]
print(f"Probability of transitioning from E to C: {prob_E_to_C:.4f}")


### PART E) Get the number of steps to see convergence ###
# Initialize a uniform probability distribution
x = np.ones(n) / n
tMatrix_power = transition_matrix.copy()

print("\nPower iteration results:")
history = []
for step in range(10):  # limit to 10 steps max
    x_new = x.dot(tMatrix_power)
    print(f"Step {step}: {x_new}")
    history.append(x_new)
    
    #Check for convergence
    if step > 0 and np.allclose(history[-1], history[-2], atol=1e-6):
       break
    
    # Accelerate convergence: square the transition matrix
    tMatrix_power = tMatrix_power.dot(tMatrix_power)

### PART F & G) Get the LEAST central Node ###
convergence_step = step + 1
final_distribution = x_new
least_central_node = nodes[np.argmin(final_distribution)]

print(f"\nConvergence identified at step: {convergence_step}") # -1 because we start at 0

print("Final PageRank distribution:")
for node, pr in zip(nodes, final_distribution):
    print(f"{node}: {pr:.4f}")

print(f"\nLeast central node: {least_central_node}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q2.b Student Answers:
# MAGIC > __a)__ Highlight the correct answer (Add ** before and after your selected choice):
# MAGIC * [0, 0, 0]
# MAGIC * **[0.16666667, 0.02777778, 0.0007716]**
# MAGIC * [0.16666667, 0.16666667, 0.16666667]
# MAGIC * [0.16666667, 0, 0.16666667]
# MAGIC
# MAGIC > __b)__ Highlight the correct answer (Add ** before and after your selected choice):
# MAGIC * Each column should represent a probability distribution but these don't sum to 1. In fact the more iterations we run, the higher their sum is.
# MAGIC * **Each column should represent a probability distribution but these don't sum to 1. In fact the more iterations we run, the lower their sum is.**
# MAGIC
# MAGIC
# MAGIC > __c)__ Highlight the correct answer (Add ** before and after your selected choice):
# MAGIC * A dangling node is a node with outlinks but no inlinks. We need to redistribute the dangling mass to maintain stochasticity.
# MAGIC * A dangling node is a node with no outlinks or inlinks. We need to redistribute the dangling mass to maintain stochasticity.
# MAGIC * **A dangling node is a node with inlinks but no outlinks. We need to redistribute the dangling mass to maintain stochasticity.**
# MAGIC
# MAGIC > __d)__ Highlight the correct answer (Add ** before and after your selected choice):
# MAGIC * **Irreducibility: There must be a sequence of transitions of non-zero probability from any state to any other. In other words, the graph has to be connected (a path exists from all vertices to all vertices). No, the webgraph is not irreducible as it will have disconnected segments.**
# MAGIC * Irreducibility: There must be a sequence of transitions of non-zero probability from any state to any other. In other words, the graph has to be connected (a path exists from all vertices to all vertices). Yes, the webgraph is irreducible as all segments are connected.
# MAGIC
# MAGIC > __e)__ Highlight the correct answer (Add ** before and after your selected choice):
# MAGIC * Aperiodicity: States are not partitioned into sets such that all state transitions occur cyclicly from one set to another. No, the web graph is not aperiodic as all links are outlinks.
# MAGIC * **Aperiodicity: States are not partitioned into sets such that all state transitions occur cyclicly from one set to another. Yes, the web graph is aperiodic as all we need is a single page with a link back to itself. An anchor link is one such link.**
# MAGIC
# MAGIC > __f)__ Highlight the correct answers - 2 options (Add ** before and after your selected choice):
# MAGIC * **teleportation**
# MAGIC * oversampling
# MAGIC * dampening
# MAGIC * **deterministic random surfer behavior**

# COMMAND ----------

# part 2.a.d - recall what the adjacency matrix looked like (RUN THIS CELL AS IS)
TOY_ADJ_MATR

# COMMAND ----------

# part 2.a.d - use TOY_ADJ_MATR to create a right stochastic transition matrix for this graph
################ YOUR CODE HERE #################
transition_matrix = TOY_ADJ_MATR.div(TOY_ADJ_MATR.sum(axis=1), axis=0) # replace with your code

################ (END) YOUR CODE #################
print(transition_matrix)

# COMMAND ----------

# part 2.a.e - compute the steady state using the transition matrix 
def power_iteration(xInit, tMatrix, nIter, verbose = True):
    """
    Function to perform the specified number of power iteration steps to 
    compute the steady state probability distribution for the given
    transition matrix.
    
    Args:
        xInit     - (n x 1 array) representing inial state
        tMatrix  - (n x n array) transition probabilities
        nIter     - (int) number of iterations
    Returns:
        state_vector - (n x 1 array) representing probability 
                        distribution over states after nSteps.
    
    NOTE: if the 'verbose' flag is on, your function should print the step
    number and the current matrix at each iteration.
    """
    state_vector = None
    ################ YOUR CODE HERE #################
    state_vector = xInit
    for ix in range(nIter):    
        
        tMatrix = tMatrix@tMatrix
        new_state_vector = state_vector@tMatrix
        state_vector = new_state_vector
        
        if verbose:
            print(f'Step {ix}: {state_vector}, sum: {np.sum(state_vector)}')
    
    ################ (END) YOUR CODE #################
    return state_vector

# COMMAND ----------

# part 2.a.e - run 10 steps of the power_iteration (RUN THIS CELL AS IS)
xInit = np.array([1.0, 0, 0, 0, 0]) # note that this initial state will not affect the convergence states
states = power_iteration(xInit, transition_matrix, 10, verbose = True)

# COMMAND ----------

# MAGIC %md
# MAGIC __`Expected Output for part 2.a.e after 10 iterations:`__  
# MAGIC >Steady State Probabilities:
# MAGIC ```
# MAGIC [0.10526316 0.15789474 0.18421053 0.23684211 0.31578947] 
# MAGIC ```

# COMMAND ----------

# part 2.b.a - run this code to create a second toy graph (RUN THIS CELL AS IS)
TOY2_GRAPH = {'nodes':['A', 'B', 'C', 'D', 'E'],
              'edges':[('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'D'), 
                       ('B', 'E'), ('C', 'A'), ('C', 'E'), ('D', 'B')]}

# COMMAND ----------

# part 2.b.a - simple visualization of our test graph using nx (RUN THIS CELL AS IS)
G = nx.DiGraph()
G.add_nodes_from(TOY2_GRAPH['nodes'])
G.add_edges_from(TOY2_GRAPH['edges'])
display(nx.draw_networkx(G, pos=nx.circular_layout(G), with_labels=True, alpha = 0.5))

# COMMAND ----------

# part 2.b.a - run 10 steps of the power iteration method here
# HINT: feel free to use the functions get_adj_matr() and power_iteration() you wrote above
################ YOUR CODE HERE #################
adj_matr = get_adj_matr(TOY2_GRAPH).to_numpy()
#calculate the out degrees for each node
divisors = [1/x if x != 0 else 0 for x in np.sum(adj_matr, axis=1)]
D = np.diag(divisors)
#divide each row in the adjacency matrix by it's number of out degrees
trans_matr =  D@adj_matr
print(adj_matr)
print(trans_matr)
xInit = np.array([1.0, 0, 0, 0, 0]) # note that this initial state will not affect the convergence states
#use power iteration function to get state vectors
state = power_iteration(xInit, trans_matr, 10, verbose = True)
################ (END) YOUR CODE #################

# COMMAND ----------

# MAGIC %md
# MAGIC # Question 3: Data and EDA
# MAGIC # Author: 3a)-Hildah
# MAGIC # Author: 3b)-Sohail
# MAGIC
# MAGIC ## About the Data
# MAGIC The main dataset for this data consists of a subset of a 500GB dataset released by AWS in 2009. The data includes the source and metadata for all of the Wikimedia wikis. You can read more here: 
# MAGIC > https://aws.amazon.com/blogs/aws/new-public-data-set-wikipedia-xml-data. 
# MAGIC
# MAGIC Use the cells below to download the wikipedia data and a test file for use in developing your PageRank implementation (note that we'll use the 'indexed out' version of the graph) and to take a look at the files.
# MAGIC
# MAGIC ## a. EDA - Number of nodes:
# MAGIC As usual, before we dive in to the main analysis, we'll peform some exploratory data anlysis to understand our dataset. Please use the test graph that you downloaded to test all your code before running the full dataset.
# MAGIC
# MAGIC ### Q3a. Tasks:
# MAGIC * __a) Fill in the blanks:__ What is the format of the raw data? What does the first value represent? What does the second part of each line represent? [__`HINT:`__ _no need to go digging here, just visually inspect the outputs of the head commands that we ran after loading the data above._]
# MAGIC * __b) Multiple Choice:__ Run the provided bash command to count the number of records in the raw dataset. Explain why this is _not_ the same as the number of total nodes in the graph.
# MAGIC * __c) Code:__ In the space provided below write a Spark job to count the _total number_ of nodes in this graph. 
# MAGIC * __d) Numerical Answer:__ How many dangling nodes are there in this wikipedia graph? [__`HINT:`__ _you should not need any code to answer this question._]
# MAGIC
# MAGIC ## b. EDA - Out-Degree distribution:
# MAGIC As you've seen in previous homeworks the computational complexity of an implementation depends not only on the number of records in the original dataset but also on the number of records we create and shuffle in our intermediate representation of the data. The number of intermediate records required to update PageRank is related to the number of edges in the graph. In this question you'll compute the average number of hyperlinks on each page in this data and visualize a distribution for these counts (the out-degree of the nodes). 
# MAGIC
# MAGIC ### Q3b. Tasks:
# MAGIC * __a) Code:__ In the space provided below write a Spark job to stream over the data and compute all of the following information:
# MAGIC   * Count the number of out-degree for each __non-dangling node__ and return the names of the top 10 pages with the most hyperlinks
# MAGIC   * Find the average out-degree for all __non-dangling nodes__ in the graph
# MAGIC   * Take a 1000 point sample of these out-degree counts and plot a histogram of the result. 
# MAGIC * __b) Numerical Answer:__ What is the average out-degree of the `testRDD`?
# MAGIC * __c) Multiple Choice:__ What is the top node by out-degree of the `wikiRDD`?
# MAGIC * __d) Fill in the blanks:__ In the context of the PageRank algorithm, how is information about a node's out degree used?
# MAGIC * __e) Multiple Choices:__ What does it mean if a node's out-degree is 0? In PageRank how will we handle these nodes differently than others? Select all that apply. [__Hint:__ _select 3 answers_]
# MAGIC
# MAGIC __`NOTE:`__ Please observe scalability best practices in the design of your code & comment your work clearly. You will be graded on both the clarity and the design.

# COMMAND ----------

# open test_graph.txt file to see format (RUN THIS CELL AS IS)
with open('/dbfs/mnt/mids-w261/HW5/HW5/test_graph.txt', "r") as f_read:
  for line in f_read:
    print(line)

# COMMAND ----------

# load the data into Spark RDDs for convenience of use later (RUN THIS CELL AS IS)
DATA_PATH = 'dbfs:/mnt/mids-w261/HW5/HW5/'
testRDD = sc.textFile(DATA_PATH +'test_graph.txt')
indexRDD = sc.textFile(DATA_PATH + '/indices.txt')
wikiRDD = sc.textFile(DATA_PATH + '/all-pages-indexed-out.txt')

# COMMAND ----------

# display testRDD (RUN THIS CELL AS IS)
testRDD.take(10)

# COMMAND ----------

# display indexRDD (RUN THIS CELL AS IS)
indexRDD.take(10)

# COMMAND ----------

# display wikiRDD (RUN THIS CELL AS IS)
wikiRDD.take(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q3.a Student Answers:
# MAGIC > __a)__ Fill in the blanks with any of these possible statements based on the following category.
# MAGIC   > * type = [a dictionary, **an adjacency list**]
# MAGIC   > * value = [**node id**, node count]
# MAGIC   > * part = [a tuple of out-edges, **a dictionary of linked pages (neighbor nodes) and the number of times that page is linked**]
# MAGIC * The raw data is in the format of [an adjacency list].
# MAGIC * The first value of each line represents [node id].
# MAGIC * The second part of each line represents [a dictionary of linked pages (neighbor nodes) and the number of times that page is linked].
# MAGIC
# MAGIC > __b)__ Highlight the correct answer (Add ** before and after your selected choice):
# MAGIC * Webpages (i.e. nodes) that are not referenced by a hyperlink (i.e. in-edges) won't have a record in this raw representation of the graph.
# MAGIC
# MAGIC * **__Webpages (i.e. nodes) that don't have any hyperlinks (i.e. out-edges) won't have a record in this raw representation of the graph.__**
# MAGIC
# MAGIC > __d)__ Numeric Answer: 11
# MAGIC
# MAGIC ### Q3.b Student Answers:
# MAGIC > __b)__ Numeric Answer: 1.7
# MAGIC
# MAGIC > __c)__ Highlight the correct answer (Add ** before and after your selected choice):
# MAGIC
# MAGIC * **__'7883280'__**
# MAGIC * '7858931'
# MAGIC * '7804599'
# MAGIC * '7705822'
# MAGIC * '11185362'
# MAGIC * '5760310'  
# MAGIC
# MAGIC > __d)__ Fill in the blanks with any of these possible statements based on the following category.
# MAGIC   > * value = [probability of random jump, total inbound links, **__current rank (or current mass)__**]
# MAGIC   > * node = [**__each of its neighbor(s)__**, every node in the network]
# MAGIC
# MAGIC   * The outdegree is used in calculating each node's "contribution" to its neighbors upon each iteration/update step. Specifically we take a node's [value], divide it by the out-degree, and then redistribute that value to [node] to get added up.
# MAGIC
# MAGIC > __e)__ Highlight the correct answers (Add ** before and after your selected choice):
# MAGIC
# MAGIC * **__Nodes with out-degree 0 are dangling nodes (also known as 'sinks'). __**
# MAGIC * A node with out-degree 0 has a full adjacency list which contains all other nodes in the network.
# MAGIC * **__Without modification of distribution, the mass from the nodes with out-degree 0 is not distributed to other nodes. The total rank (or accumulated mass) from all nodes will be less than 1, and using Markov chain will not converge.__**
# MAGIC * Without modification of distribution, the mass from the nodes with out-degree 0 is randomly distributed to other nodes using teleport vector.
# MAGIC * **__In PageRank, we redistribute the mass from nodes with out-degree 0 evenly across the rest of the graph in PageRank.__**

# COMMAND ----------

# part 3.a.b - count the number of records in the raw data (RUN THIS CELL AS IS)
# 5781290 - Approx. time: 20.55 seconds
print(wikiRDD.count())

# COMMAND ----------

# part 3.a.c - write your Spark job here (compute total number of nodes)
def count_nodes(dataRDD):
    """
    Spark job to count the total number of nodes.
    Returns: integer count 
    """    
    ############## YOUR CODE HERE ###############
    src_nodes = dataRDD.map(lambda line: line.split('\t')[0])
    
    # Parse each line and extract destination nodes (keys from dictionary)
    dest_nodes = dataRDD.flatMap(lambda line: eval(line.split('\t')[1]).keys())
    
    # Union both and count distinct
    totalCount = src_nodes.union(dest_nodes).distinct().count()

    # totalCount = dataRDD.map(lambda line: line.split('\t')[0]).distinct().count() 
    # + dataRDD.flatMap(lambda line: eval(line.split('\t')[1]).keys()).distinct().count() # replace with your code 
    
    ############## (END) YOUR CODE ###############   
    return totalCount

# COMMAND ----------

# part 3.a.c - run your counting job on the test file (RUN THIS CELL AS IS)
# Approx time: 0.41 seconds
tot = count_nodes(testRDD)
print(f'Total Nodes: {tot}')

# COMMAND ----------

# part 3.a.c - run your counting job on the full file (RUN THIS CELL AS IS)
# Approx time: 6.35 minutes
tot = count_nodes(wikiRDD)
print(f'Total Nodes: {tot}')

# COMMAND ----------

# part 3.a.d - number of dangling nodes - Use your previous results! You can hard code the number using arithmetic of previous results
############## YOUR CODE HERE ###############
num_dangling = 9410987 # your code here
############## (END) YOUR CODE ###############   

# RUN AS IS
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

data = [5781290, num_dangling]
labels = ["regular", "dangling"]

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n{:d}".format(pct, absolute)

wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"))
ax.legend(wedges, labels,
          title="Node Type",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=12, weight="bold")
ax.set_title("Ratio of dangling nodes")
display(plt.show())

# COMMAND ----------

# part 3.b.a - write your Spark job here (compute average in-degree, etc)
def count_degree(dataRDD, n):
    """
    Function to analyze out-degree of nodes in a a graph.
    Returns: 
        top  - (list of 10 tuples) nodes with most edges
        avgDegree - (float) average out-degree for non-dangling nodes
        sampledCounts - (list of integers) out-degree for n randomly sampled non-dangling nodes
    """
    # helper func
    def parse(line):
        node, edges = line.split('\t')
        return (node, ast.literal_eval(edges))
    
    ############## YOUR CODE HERE ###############

    # need a count of non dangling for their average
    parsed = dataRDD.map(parse)
    non_dangling = parsed.filter(lambda x: len(x[1]) > 0)
    degrees = non_dangling.map(lambda x: (x[0], len(x[1])))
    

    top = degrees.takeOrdered(10, key=lambda x: -x[1])  # replace with your code
    avgDegree = degrees.map(lambda x: x[1]).sum()/degrees.count()  # replace with your code
    sampledCounts = degrees.takeSample(False, n)  # replace with your code
    sampledCounts = [x[1] for x in sampledCounts]
    ############## (END) YOUR CODE ###############
    
    return top, avgDegree, sampledCounts

# COMMAND ----------

# part 3.b.b - run your job on the test file (RUN THIS CELL AS IS)
# Approx time: 1.20 seconds
test_results = count_degree(testRDD,10)
print("Average out-degree: ", test_results[1])
print("Top 10 nodes (by out-degree:)\n", test_results[0])

# COMMAND ----------

# part 3.b.b - plot results from test file (RUN THIS CELL AS IS)
plt.hist(test_results[2], bins=10)
plt.title("Distribution of Out-Degree")
display(plt.show())

# COMMAND ----------

# part 3.b.c - run your job on the full file (RUN THIS CELL AS IS)
# Approx time: 3.69 minutes
full_results = count_degree(wikiRDD,1000)
print("Average out-degree: ", full_results[1])
print("Top 10 nodes (by out-degree:)\n", full_results[0])

# COMMAND ----------

# part 3.b.c - plot results from full file (RUN THIS CELL AS IS)
plt.hist(full_results[2], bins=50)
plt.title("Distribution of Out-Degree")
display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC # Question 4: PageRank
# MAGIC # Author: 4a)Hildah
# MAGIC # Author: 4b)Roz
# MAGIC
# MAGIC ## a. Initialize the Graph
# MAGIC One of the challenges of performing distributed graph computation is that you must pass the entire graph structure through each iteration of your algorithm. As usual, we seek to design our computation so that as much work as possible can be done using the contents of a single record. In the case of PageRank, we'll need each record to include a node, its list of neighbors and its (current) rank. In this question you'll initialize the graph by creating a record for each dangling node and by setting the initial rank to 1/N for all nodes. 
# MAGIC
# MAGIC __`NOTE:`__ Your solution should _not_ hard code \\(N\\).
# MAGIC
# MAGIC ### Q4.a Tasks:
# MAGIC * __a) Multiple Choice:__ What is \\(N\\)? 
# MAGIC
# MAGIC * __b) Multiple Choices:__ Using the analogy of the infinite random web-surfer, how do we use \\(\frac{1}{N}\\)? [__HINT:__ _select 2 answers_]
# MAGIC
# MAGIC * __c) Code:__ Fill in the missing code below to create a Spark job that:
# MAGIC   * parses each input record
# MAGIC   * creates a new record for any dangling nodes and sets it edges (neighbors) to be a empty [__HINT:__ _you should use the same data type with the edges for non-dangling nodes_]
# MAGIC   * initializes a rank of 1/N for each node
# MAGIC   * returns a pair RDD with records in the format specified by the docstring
# MAGIC
# MAGIC * __d) Numerical Answer:__ Run the provided code to confirm that your job in `part c` has a record for each node. Your records should match the format specified in the docstring and the count should match what you computed in question 3a. Then answer the question: how many edges does the node `13415942` have? [__`TIP:`__ _you might want to take a moment to write out what expected outputs you should get for the test graph, this will help you know your code works as expected_]
# MAGIC  
# MAGIC __`NOTE:`__ Please observe scalability best practices in the design of your code & comment your work clearly. You will be graded on both the clarity and the design.
# MAGIC
# MAGIC
# MAGIC ## b. Iterate until Convergence
# MAGIC Finally we're ready to compute the page rank. In this last question you'll write a Spark job that iterates over the initialized graph updating each nodes score until it reaches a convergence threshold. The diagram below gives a visual overview of the process using a 5 node toy graph. Pay particular attention to what happens to the dangling mass at each iteration.
# MAGIC
# MAGIC <img src='https://github.com/kyleiwaniec/w261_assets/blob/master/images/HW5/PR-illustrated.png?raw=true' width=50%>
# MAGIC
# MAGIC __`A Note about Notation:`__ The formula above describes how to compute the updated page rank for a node in the graph. The \\(P\\) on the left hand side of the equation is the new score, and the \\(P\\) on the right hand side of the equation represents the accumulated mass that was re-distributed from all of that node's in-links. Finally, \\(|G|\\) is the number of nodes in the graph (which we've elsewhere refered to as \\(N\\)).
# MAGIC
# MAGIC ### Q4.b Tasks:
# MAGIC * __a) Multiple Choices:__ In terms of the infinite random walk analogy, interpret the meaning of the first term in the PageRank calculation: \\(\alpha * \frac{1}{|G|}\\). 
# MAGIC     [__Hint__: _select two answers_]
# MAGIC
# MAGIC * __b) Multiple Choice:__ In the equation for the PageRank calculation above what does \\(m\\) represent and why do we divide it by \\(|G|\\)?
# MAGIC
# MAGIC * __c) Numerical Answer:__ Keeping track of the total probability mass after each update is a good way to confirm that your algorithm is on track. How much should the total mass be after each iteration?
# MAGIC
# MAGIC * __d) Code:__ Fill in the missing code below to create a Spark job that take the initialized graph as its input then iterates over the graph and for each pass:
# MAGIC   * reads in each record and redistributes the node's current score to each of its neighbors
# MAGIC   * uses an accumulator to add up the dangling node mass and redistribute it among all the nodes. (_Don't forget to reset this accumulator after each iteration!_)
# MAGIC   * uses an accumulator to keep track of the total mass being redistributed.( _This is just for your own check, its not part of the PageRank calculation. Don't forget to reset this accumulator after each iteration._)
# MAGIC   * aggregates these partial scores for each node
# MAGIC   * applies telportation and damping factors as described in the formula above.
# MAGIC   * combine all of the above to compute the PageRank as described by the formula above.
# MAGIC
# MAGIC * __e) Multiple Choice:__  What is the top ranked node of the `wikiaGraphRDD`?
# MAGIC     
# MAGIC > TIP: You can check your work for part `d` by looking at the nodes in the top 20-40th positions by PageRank score, which should match the expected output.
# MAGIC
# MAGIC __WARNING:__ Some pages contain multiple hyperlinks to the same destination, please take this into account when redistributing the mass.
# MAGIC
# MAGIC __NOTE:__ Please observe scalability best practices in the design of your code & comment your work clearly. You will be graded on both the clarity and the design.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q4.a Student Answers:
# MAGIC > __a)__ Highlight the correct answer (Add ** before and after your selected choice):
# MAGIC * The rank for each edge
# MAGIC * The total number of edges in the graph
# MAGIC * The rank for each node
# MAGIC
# MAGIC * **__The total number of nodes in the graph__**
# MAGIC * The total number of non-dangling nodes in the graph
# MAGIC
# MAGIC > __b)__ Highlight the correct answers (Add ** before and after your selected choice):
# MAGIC
# MAGIC * **__To initialize each node's rank__**
# MAGIC * To identify where each node should converge to
# MAGIC * To suggest that the random surfer is equally likely to end her random walk anywhere on the graph
# MAGIC * **__To suggest that the random surfer is equally likely to start her random walk anywhere on the graph__**
# MAGIC
# MAGIC > __d)__ Numeric Answer: __170___
# MAGIC
# MAGIC ### Q4.b Student Answers:
# MAGIC > __a)__ Highlight the correct answers (Add ** before and after your selected choice):
# MAGIC
# MAGIC * alpha is the number of nodes, with an equal probability of being "jumped to".
# MAGIC * **\\(|G|\\) is the number of nodes, with an equal probability of being "jumped to".**
# MAGIC * **alpha is the teleportation factor (i.e. the probability of a random jump)**
# MAGIC * \\(|G|\\)  is the teleportation factor (i.e. the probability of a random jump)
# MAGIC
# MAGIC > __b)__ Highlight the correct answer (Add ** before and after your selected choice):
# MAGIC
# MAGIC * m is the total mass of all nodes which we distribute to all |G| nodes equally.
# MAGIC * **m is the dangling mass which we distribute to all |G| nodes equally.**
# MAGIC
# MAGIC > __c)__ Numeric Answer: __1___
# MAGIC
# MAGIC > __e)__ Highlight the correct answer (Add ** before and after your selected choice):
# MAGIC
# MAGIC * 1184351
# MAGIC * **__13455888__**
# MAGIC * 4695850
# MAGIC * 5051368
# MAGIC * 2437837

# COMMAND ----------

# part 4.a.c - job to initialize the graph
def initGraph(dataRDD):
    """
    Spark job to read in the raw data and initialize an 
    adjacency list representation with a record for each
    node (including dangling nodes).
    
    Returns: 
        graphRDD -  a pair RDD of (node_id , (score, edges))
        
    NOTE: The score should be a float, but you may want to be 
    strategic about how format the edges... there are a few 
    options that can work. Make sure that whatever you choose
    is sufficient for Question 8 where you'll run PageRank.
    """
    ############## YOUR CODE HERE ###############

    # write any helper functions here

    def parse_line(line):
        """Parse a line into node_id and adjacency dictionary"""
        parts = line.split('\t')
        node_id = parts[0]
        # Convert to dict
        adj_dict = eval(parts[1])  
        return (node_id, adj_dict)
    
    def dict_to_list(adj_dict):
        """Convert adjacency dictionary to list of neighbors for PageRank"""
        neighbors = []
        for neighbor, count in adj_dict.items():
            # Add neighbor 'count' times to represent multiple edges
            neighbors.extend([neighbor] * count)
        return ','.join(neighbors)
    
    # write your main Spark code here
    # Parse the input data
    parsed_data = dataRDD.map(parse_line)
    
    # Get unique src and dest nodes
    src_nodes = parsed_data.map(lambda x: int(x[0]))
    dest_nodes = parsed_data.flatMap(lambda x: [int(k) for k in x[1].keys()])
    all_nodes = src_nodes.union(dest_nodes).distinct()
    
    #Calculate N (total number of nodes)
    N = all_nodes.count()
    initial_score = 1.0 / N
    
    # Create records for nodes with outgoing edges
    nodes_with_edges = parsed_data.map(lambda x: (int(x[0]), (initial_score, dict_to_list(x[1]))))
    
    # Find dangling nodes
    existing_source_nodes = nodes_with_edges.map(lambda x: int(x[0]))
    dangling_nodes = all_nodes.subtract(existing_source_nodes)
    
    # Create records for dangling nodes with empty edge lists
    dangling_records = dangling_nodes.map(lambda node: (node, (initial_score, [])))
    
    # Step 7: Union both RDDs to get the complete graph
    graphRDD = nodes_with_edges.union(dangling_records) # replace with your code

    
    ############## (END) YOUR CODE ##############
    
    return graphRDD

# COMMAND ----------

#delete
wikiRDD.take(10)

# COMMAND ----------

# part 4.a.d - run your Spark job on the test graph (RUN THIS CELL AS IS)
# Approx time: 1.55 seconds
testGraph = initGraph(testRDD).collect()
testGraph

# COMMAND ----------

# part 4.a.d - run your code on the main graph (RUN THIS CELL AS IS)
# Approx time: 8.16 minutes
start = time.time()
wikiGraphRDD = initGraph(wikiRDD)
print(f'... full graph initialized in {time.time() - start} seconds')

# COMMAND ----------

# part 4.a.d - confirm record format and count (RUN THIS CELL AS IS)
# Approx time: 2.40 minutes
start = time.time()
print(f'Total number of records: {wikiGraphRDD.count()}')
node = wikiGraphRDD.sortByKey().lookup(13415942) 
print("Node 13415942's initialized record:", node)
print(f"\n\nNode 13415942 has {len(node[0][1].split(','))} outlinks.")
print(f'Time to complete: {time.time() - start} seconds.')

# COMMAND ----------

# part 4.b.d - provided FloatAccumulator class (RUN THIS CELL AS IS)
from pyspark.accumulators import AccumulatorParam
class FloatAccumulatorParam(AccumulatorParam):
    """
    Custom accumulator for use in page rank to keep track of various masses.
    
    IMPORTANT: accumulators should only be called inside actions to avoid duplication.
    We stringly recommend you use the 'foreach' action in your implementation below.
    """
    def zero(self, value):
        return value
    def addInPlace(self, val1, val2):
        return val1 + val2

# COMMAND ----------

# part 4.b.d - job to run PageRank
def runPageRank(graphInitRDD, alpha = 0.15, maxIter = 10, verbose = True):
    """
    Spark job to implement page rank
    Args: 
        graphInitRDD  - pair RDD of (node_id , (score, edges))
                      - i.e. [(5, (0.09090909090909091, '4,4,4,2,6')), ... ]
        alpha         - (float) teleportation factor
        maxIter       - (int) stopping criteria (number of iterations)
        verbose       - (bool) option to print logging info after each iteration
    Returns:
        steadyStateRDD - pair RDD of (node_id, pageRank)
    """
    # teleportation:
    a = sc.broadcast(alpha)
    
    # damping factor:
    d = sc.broadcast(1-a.value)
    
    # initialize accumulators for dangling mass & total mass
    mmAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    totAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    
    ############## YOUR CODE HERE ###############
    
    # write your helper functions here, 
    # please document the purpose of each clearly 
    # for reference, the master solution has 5 helper functions.
    
    def distribute_non_dangling_mass(node_id, score, neighbors):
        """
        Distribute PageRank mass to non dangling neighbors.
        Args:
            node_id (int): the source node ID
            score (float): the current PageRank score
            neighbors (str or list): the outgoing neighbors (comma-separated string or empty list)
        Emits:
           (neighbor_node_id, partial_mass) for each neighbor
        """

        outlinks = neighbors.split(',')
        mass = score / len(outlinks)
        for neighbor in outlinks:
            yield (int(neighbor), mass)

    def combine_mass(mass1, mass2):
        """
        Simple reducer for combining partial PageRank mass.
        """
        return mass1 + mass2 
    
    def update_score(old_score, incoming_mass, dangling_mass, N):
        """
        Apply teleportation and damping to incoming mass.
        """
        new_score = a.value / N + d.value * (dangling_mass / N + incoming_mass)
        totAccum.add(new_score) # only triggered once per loop
        return new_score
        
    # write your main Spark Job here (including the for loop to iterate)
    # for reference, the master solution is 21 lines including comments & whitespace
    # steadyStateRDD = None  # replace with your code

    # Get total number of nodes
    N = graphInitRDD.count()
    
    # Initialize graph and cache it for iterative access
    graph = graphInitRDD.cache()

    for i in range(maxIter):
        # Reset accumulators at the start of each iteration
        mmAccum.value = 0.0
        totAccum.value = 0.0

        # Calculate dangling mass
        graph.filter(lambda x: not x[1][1]) \
             .foreach(lambda x: mmAccum.add(x[1][0]))
        dangling_mass = mmAccum.value


        # Distribute mass to non-dangling neighbors
        scores = graph.filter(lambda x: x[1][1]) \
                      .flatMap(lambda x: distribute_non_dangling_mass(x[0], x[1][0], x[1][1]))
        agg_mass = scores.reduceByKey(combine_mass)        

        # After left join: (node_id, ((old_score, neighbors), incoming_mass))
        # i.e. (2, ((0.09, '3'), 0.12))
        #      (1, ((0.09, []), None)
        graph = graph.leftOuterJoin(agg_mass) \
                     .mapValues(lambda x: (update_score(x[0][0], x[1] if x[1] else 0.0, dangling_mass, N), x[0][1])) \
                     .cache() # Cache the new graph for the next iteration.
        
        # Trigger all transformations and populate totAccum
        graph.foreach(lambda x: None)
        
        if verbose:
            print(f"Iteration {i+1}: Total mass = {totAccum.value}, Dangling mass = {mmAccum.value}")
    steadyStateRDD = graph.mapValues(lambda x: x[0])
    ############## (END) YOUR CODE ###############
    
    return steadyStateRDD

# COMMAND ----------

# part 4.b.d - run PageRank on the test graph (RUN THIS CELL AS IS)
# Note: while developing your code you may want turn on the verbose option
# Approx time: 11.80 seconds
nIter = 20
testGraphRDD = initGraph(testRDD)
start = time.time()
test_results = runPageRank(testGraphRDD, alpha = 0.15, maxIter = nIter, verbose = True)
print(f'...trained {nIter} iterations in {time.time() - start} seconds.')
print(f'Top 20 ranked nodes:')
test_results.takeOrdered(20, key=lambda x: - x[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC __`expected results for the test graph:`__
# MAGIC ```
# MAGIC [(2, 0.3620640495978871),
# MAGIC  (3, 0.333992700474142),
# MAGIC  (5, 0.08506399429624555),
# MAGIC  (4, 0.06030963508473455),
# MAGIC  (1, 0.04255740809817991),
# MAGIC  (6, 0.03138662354831139),
# MAGIC  (8, 0.01692511778009981),
# MAGIC  (10, 0.01692511778009981),
# MAGIC  (7, 0.01692511778009981),
# MAGIC  (9, 0.01692511778009981),
# MAGIC  (11, 0.01692511778009981)]
# MAGIC ```
# MAGIC ---

# COMMAND ----------

# part 4.b.d - run PageRank on the full graph (RUN THIS CELL AS IS)
# NOTES: 
# - wikiGraphRDD should have been computed & cached above!
# - Rounding errors after 10 or so decimal places are acceptable
# - This will take a long time to run...about 1.4 hours with 2 workers.
nIter = 10
start = time.time()
full_results = runPageRank(wikiGraphRDD, alpha = 0.15, maxIter = nIter, verbose = True)
print(f'\n...trained {nIter} iterations in {time.time() - start} seconds.\n')

print(f'Top 20-40th ranked nodes:\n')
top_40 = full_results.takeOrdered(40, key=lambda x: - x[1])
# print results from 20th to 40th highest PageRank
for result in top_40[20:]:
    print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Expected output for pages in top 20-40:**
# MAGIC ```
# MAGIC (9386580, 0.0002956970419566159)
# MAGIC (15164193, 0.0002843018514565407)
# MAGIC (7576704, 0.00028070286946292773)
# MAGIC (12074312, 0.0002789390949918953)
# MAGIC (3191491, 0.0002733179890615315)
# MAGIC (2797855, 0.0002728954946638793)
# MAGIC (11253108, 0.000272424060446348)
# MAGIC (13725487, 0.00027020945537652716)
# MAGIC (2155467, 0.0002664929816794768)
# MAGIC (7835160, 0.00026125677808592145)
# MAGIC (4198751, 0.00025892998307531953)
# MAGIC (9276255, 0.0002546338645363715)
# MAGIC (10469541, 0.0002537014715361682)
# MAGIC (11147327, 0.00025154645090952246)
# MAGIC (5154210, 0.00024927029620654557)
# MAGIC (2396749, 0.0002481158912959703)
# MAGIC (3603527, 0.0002465136725657213)
# MAGIC (3069099, 0.00023478481004084842)
# MAGIC (12430985, 0.0002334928892642041)
# MAGIC (9355455, 0.00022658004696908508)
# MAGIC ```
# MAGIC  ---

# COMMAND ----------

top_40[:10]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Congratulations, you have completed the last homework for 261! 