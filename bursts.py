import sys
import math
from collections import deque

def read_commands():
    #this function reads the arguments from the command line and initializes some of our variables

    if (sys.argv[1] == 'viterbi'):
        algorithm = 'viterbi'
    else:
        algorithm = 'trellis'
        
    file = sys.argv[2]
    show_messages = False
    g = 1
    s = 2
    for i in range(3,len(sys.argv)):
        argument = sys.argv[i]

        #check to see that we are not in the last iteration of the loop
        if (i != len(sys.argv) - 1):
            next = sys.argv[i+1]
            #check to see if a next argument does exist
            if (next is not None):
                if (argument == '-s'):
                    s = next
                elif (argument == '-g'):
                    g = next
        
        if (argument == '-d'):
            show_messages = True
    return file, algorithm, s, g, show_messages


def read_file(file):
    #this function reads our txt file(graph) and saves it as a list of time intervals
    #first initialize an empty list
    intervals = []

    with open(file,'r') as f:
        for line in f:
            numbers = line.strip().split()
            for number in numbers:
                intervals.append(float(number))
    return intervals


def calculate_k_t():
    #calculate the number k that represent the sum of q instances that our system has
    #first, find the smallest interval within our list of intervals
    min = math.inf
    for i in range(1,len(x)):
      if (x[i] < min):
            min = x[i]

    #calculate T based on the expression of the assignment
    t = intervals[-1]
    #then calculate the expression from the assignment
    k = math.ceil(1 + math.log(t, float(s)) + math.log(1 / min, float(s)))
    return k,t

def create_x():
    #create the x array by copying the first column of the intervals array. we want array x to start at index 1, so at index 0, we insert 0
    x = [0]
    for i in range(1, len(intervals)):
        var = intervals[i] - intervals[i - 1]
        x.append(round(var, 2))
    return x

def calculate_l(i):
    #calculate l, as defined in the assignment. first we calculate g(called gamma, so as to be  distinguishable from the g from the command line that refers to the cost function)
    gamma =t / n if (n != 0) else print('cant divide with 0')
    if (gamma is not None):
        if (gamma != 0):
            return float(s)**i / gamma
        else:
            return None
    else:
        return None


def create_graph():
    #create a graph that will be used for our implementation of the bellman ford moore algorithm and initialize it with time zero where we have q zer0
    #first initialize a dictionary and create the graph keys-vertices
    graph = {}
    for time in range(0,n + 1):
        for q in range(0,k):
            graph.update({(time,q) : []})

    #next define the way the edges are calculated
    #first for the initial link between time = 0 and all states    
    for j in range(0,k):
        transition_cost = abs(cost(0,j))
        if f(j,0) > 0:
            emission_cost = abs(math.log(f(j,0), math.e))
        else:
            emission_cost = 0
        
        cost_list = [transition_cost, emission_cost, transition_cost + emission_cost]
        graph[(0,0)].append((1, j, cost_list))

    #and then for all other links
    for time in range(1, n):
        for q in range (0,k):
            for j in range (0,k):
                transition_cost = abs(cost(q,j))
                if f(j,time) > 0 :
                    emission_cost = abs(math.log(f(j,time), math.e))
                else:
                    emission_cost = 0
                cost_list = [transition_cost, emission_cost, transition_cost + emission_cost]

                graph[(time,q)].append((time + 1, j, cost_list))


    return graph

def create_c():
    #creates array c, as defined in the viterbi algorithm, with dimensions n+1 *k

    c = []
    for i in range(0, n+1):
        row = []
        for j in range(0,k):
            if (i == 0 and j == 0):
                row.append(0)
            else:
                row.append(math.inf)
        c.append(row)
    return c

def create_p():
    #create the p list, as defined in the viterbi algorithm
    p = []
    for i in range(0,k):
        row = []
        for j in range(0, n+1):
            row.append(0)
        p.append(row)
    
    return p


def cost(i,j):
    #defines a cost function, as defined in the assignment. the function returns the cost of the transition from the state of i to the state of j
    if (j <= i):
        return 0
    else:
        return (float(g) * (j - i)* math.log(n,math.e))


def f(s, t_previous):
    #this is the function that returns the number of emissions based on the exponential distribution of the interval x during state s

    l = calculate_l(s)
    var = math.e ** ((-1) * l * x[t_previous + 1])
    return (l * var)


def viterbi(): 
    #create the algorithm as defined in the assignment

    #first print the 0line of list c
    if (show_messages):
       print("[{}]".format(", ".join(str(c[0][z]) for z in range(k))))


    #the algorithm, as defined in the assignment
    for t in range(1, n+1):
        for s in range(0, k):

            lmin = 0
            cmin = c[t-1][0] + cost(0,s)

            for l in range(1,k):
            
                ccc = c[t-1][l] + cost(l,s)

                if (ccc < cmin):
                    cmin = ccc
                    lmin = l
            var = f(s, t-1)
            #print('var is',var)
            if (var != 0):
                c[t][s]= cmin - math.log(var, math.e)
            else:
                c[t][s]= cmin

            p[s][0:t] = p[lmin][0:t]
            p[s][t] = s

            #here is where the print statement should be placed, based on show_messages(the command -d)
        if (show_messages):
            print("[{}]".format(", ".join([f"{val:.2f}" if val != float('inf') else "inf" for val in c[t]])))

        cmin = c[n][0]
        smin = 0
        
        for s in range(1, k):
            if (c[n][s] < cmin):
                cmin = c[n][s]
                smin = s
    return p[smin]


def show_states():
    #this function used the states list(p[smin] from the viterbi algorithm), in order to print the list of states during the transition from one time interval to another. it also works with the bellman ford algorithm
    #to do that, we will be iterating over the states and x(that contains the intervals) lists at the same time
        #save the state at position 0
        start = 0.0
        end = 0

        for i in range(1,len(states)):
            if (states[i] == states[i - 1]):
                end += x[i] 
            else:
                if end != 0 :
                 print(f"{states[i-1]} [{round(start, 2)} {round(end, 2)})")
                start = end
                end += x[i]
        print(f"{states[-1]} [{round(start, 2)} {round(end, 2)})")

def find_states():
    #this function finds states using the dist and pred arrays, provided we are using the bellman ford algorithm
    states = deque()
    #find the minimum cost path among the states of the last time tunit
    time_index = list(pred.keys())[-1][0]
    state_index = list(pred.keys())[-1][1]
    min = dist[time_index, state_index]
    for state in range(0, k):
        if dist[time_index, state] < min :
            min = dist[time_index, state]
            state_index = state
    states.appendleft(state_index)
    #then, after having found the node with the minimum cost, iterate over its previous node, append its state to the states queue and repeat the process
    while (True):
        var = pred[time_index, state_index]
        if time_index != 0:
            state_index = int(var[1])
            states.appendleft(state_index)
            time_index -= 1
        else:
            break
    return states



def bellman_ford():
    #implements the bellman ford moore algorithm for the trellis graph
    #count the number of nodes and initialize the starting position which is node (0,0)
    nodes = len(graph.keys())
    source = (0,0)

    #initialize dictionaries for pred and dist instead of lists, since your graph nodes are tuples, not integers
    dist = {node: math.inf for node in graph.keys()}
    pred = {node: -1 for node in graph.keys()}
    #update the dist of the starting position
    dist.update({source : 0.0})
    
    #for all paths involving nodes
    for i in range(0, nodes - 1):
        for node in graph.keys(): 
            
            for element in graph[node]:

                adjacent_node = tuple(element[0:2])
                weight = element[2][2]                
                if dist[node] != math.inf and dist[adjacent_node] > dist[node] + weight:
                    if show_messages:
                        print(adjacent_node,' ', round(dist[adjacent_node], 2), ' -> ', round(dist[node] + weight, 2) , '  from  ', node, '  ', round(dist[node], 2),'  +  ', round(element[2][0], 2), ' + ', round(element[2][1], 2))
                    dist[adjacent_node] = dist[node] + weight
                    pred[adjacent_node] = node
    return dist, pred



#START

#first read our command line arguments
file, algorithm, s, g, show_messages = read_commands()

#then read our file and save the time intervals as a list
intervals = read_file(file)

#the number of intervals, is our number n. and n + 1 = the number of messages
n = len(intervals) - 1

#create the lists that are used in the viterbi algorithm
x = create_x()

#calculate the number of q instances that our system has.
k, t= calculate_k_t()
c = create_c()
p = create_p()

#execute the algorithm, viterbi, or bellman ford, accordingly
if (algorithm == 'viterbi'):    
    states =viterbi()
    #check to see if we have given the -d command in order to print messages
    if (show_messages):
        print(len(states), states)
elif(algorithm == 'trellis'):
    #first create a trellis graph
    graph = create_graph()
    dist, pred = bellman_ford()

    #find the states according to the bellman ford algorithm    
    states = find_states()

    if show_messages:
        print(len(states), [i for i in states])
#call function in order to print the list of states and time transitions
show_states()