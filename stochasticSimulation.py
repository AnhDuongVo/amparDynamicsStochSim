# stochastic simulation of AMPA receptor dynamics using the Gillespie algorithm

# This program was used to create Fig. 4.3(a),(b) and the data for 4.1, 4.2 of the manuscript.
# It plots w_i of synapses during an example simulation.
# Furthermore it plots the CV  of w_i of 7 synapses. 
# For combining the results from multiple simulations into one figure (like Fig.4.1, 4.2) please run coefficientOfVariation.py.

import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from copy import deepcopy
import time

###################################################
######## Functions ################################
###################################################

def next_values(a0,a,r1,r2):
    """returns values for the next reaction like time difference and reaction according to Gillespie"""
    
    # calculate next time
    new_time_difference = (1/a0)*np.log(1/r1)
    
    # choose next reaction R_mu under the condition that
    # sum(a[i], i=0, mu-1) < r2*a0 <= sum(a[i], i=0, mu)
    mu = 0
    N = r2*a0 - a[mu] 
    
    while N > 0:
        mu += 1
        N = N - a[mu]
    
    return(new_time_difference, mu)
    
def calculate_hi(n,m):
    """calculates the hi with the help of binomial coeff and factorial() 
    since hi is defined as total number of distinct 
    combinations of Ri reactant molecules"""
    
    b=[0]*(n+1)
    b[0]=1
    for i in range(1,n+1):
        b[i]=1
        j=i-1
        while j>0:
            b[j]+=b[j-1]
            j-=1
    hi = b[m]*factorial(m)
    return(hi)
    
def reactions_stoch(reactions):
    """gets a string of several reactions and outputs the stoichiometry array
    of substrates and products
    
    the form of the string is like '2X1+X2->X3,X1->0,X3+4X2->12X1'
    to define N species please use X1, X2, X3,..., XN-1, XN 
    only use symbols like '->' and '+', dont use spaces
    """
    
    # important variables
    substrates = [] # string reactions is splitted into substrates and products
    products = []
    sub_without_plus = [] # each subtrate is separated by '+'
    prod_without_plus = []
    sub_number_reaction = [] # list with items like [number of species, name of species]
    prod_number_reaction = []
    total_number_species = 0 # number of all species to create array
    
    # remove symobls like '+', '->', ',' and split it into substrates and products 
    one_reaction = reactions.split(",")
    for i in one_reaction:
        s_p = i.split("->")
        substrates.append(s_p[0])
        products.append(s_p[1])

    for i in substrates:
        sub_without_plus.append(i.split("+"))
    for i in products:
        prod_without_plus.append(i.split("+"))
    
    # split each item into [number of species, name of species]
    for i in sub_without_plus:
        sub_one_reaction = []
        for j in i:
            number_reaction = []
            j = j.split("X")
            if j[0].isdigit():
                number_reaction.append(int(j[0]))
            else:
                number_reaction.append(1)
            if len(j) == 2:
                number_reaction.append(int(j[1]))
                if int(j[1]) > total_number_species:
                    total_number_species = int(j[1])
            if len(number_reaction) == 2:
                sub_one_reaction.append(number_reaction)
        if len(sub_one_reaction) >= 1:
            sub_number_reaction.append(sub_one_reaction)
    
    for i in prod_without_plus:
        prod_one_reaction = []
        for j in i:
            number_reaction = []
            j = j.split("X")
            if j[0].isdigit():
                number_reaction.append(int(j[0]))
            else:
                number_reaction.append(1)
            if len(j) == 2:
                number_reaction.append(int(j[1]))
                if int(j[1]) > total_number_species:
                    total_number_species = int(j[1])
            if len(number_reaction) == 2:
                prod_one_reaction.append(number_reaction)
        if len(prod_one_reaction) >= 1:
            prod_number_reaction.append(prod_one_reaction)
    
    # create arrays for the stoichiometry of substrates and products
    sub_stoch = np.zeros((len(one_reaction), total_number_species), int)
    prod_stoch = np.zeros((len(one_reaction), total_number_species), int)
    
    # fill the arrays with the number of species
    for reaction in range(len(sub_number_reaction)):
        for species in sub_number_reaction[reaction]:
            sub_stoch[reaction][species[1]-1] = int(-species[0])
    
    for reaction in range(len(prod_number_reaction)):
        for species in prod_number_reaction[reaction]:
            prod_stoch[reaction][species[1]-1] = int(species[0])
    return(sub_stoch, prod_stoch)

###################################################
######## Gillespie algorithm ######################
###################################################
    
def gillespie_algo(s_i, init, rates, sub_stoch, prod_stoch, tmax, n_max):
    """generates a statistically correct trajectory of a stochastic equation

    input:
    s_i = array([s1,...,sN]) number of slots
    init = array([w1,...,wN,e1,...,eN,p]) number of molecules of each species
    rates = array([c1,..cM]) rates of each reaction
    sub_stoch, prod_stoch = stochiometry of substrates and products in matrix form
    tmax = maximum time
    n_max = estimated maximum number of reactions]

    output:
    store_time = array([[t1],[t2],[t3],...]) current time of each intervall
    store_number_molecules = array([[number molecules reaction 0],[number molecules reaction 0],..])
    coefficient_variation = array([CV_1,...,CV_N]) average of coefficient of variation of each synapse
    """
    
    # ****************************   
    # step 0: initialisation
    # ****************************

    # generate a array of two random numbers for step 2
    r1 = np.random.random_sample(n_max)
    r2 = np.random.random_sample(n_max)

    # initialise constant parameters
    stoch = sub_stoch + prod_stoch 
    number_reactions = np.shape(stoch)[0] # number of reactions
    number_species = np.shape(stoch)[1] # number of species
    number_synapses = int((len(init)-1)/2)

    # initialise current parameters
    current_time = 0
    current_species = init # current number of molecules of each species
    n_counter = 0 # number of already occured reactions
    
    # initialise variables to store time and molecule numbers
    store_time = np.zeros(n_max)
    store_time[n_counter] = current_time
    store_number_molecules = np.zeros((n_max, number_species))
    store_number_molecules[n_counter,:] = current_species 
    store_time_difference = np.zeros((n_max,number_synapses)) 
    store_filling_fraction_av = np.zeros(number_synapses)
    
    while (current_time < tmax) and (n_counter < n_max-1):
        
        # ****************************   
        # step 1: calculate ai and a0
        # ****************************   

        a = np.ones((number_reactions,1))

        for i in range(number_reactions):
            hi = 1  # h1 is defined as the number of distinct 
                    # combinations of Ri reactant molecules 
            for j in range(number_species):
                # check whether the reactant is involved in this reaction
                if sub_stoch[i,j] == 0:
                    continue
                else:
                    # check the reactant has molecules available
                    if current_species[j] <= 0: 
                        hi = 0
                        continue
                    else:
                        hi *= calculate_hi\
                        (int(current_species[j]),np.absolute(sub_stoch[i,j]))
                        
            a[i] = hi*rates[i]
            
        a0 = sum(a)

        # ****************************   
        # step 2: calculate the next time difference and reaction
        # ****************************   
        new_time_difference,next_r = next_values(a0,a,r1[n_counter],r2[n_counter])
        store_time_difference[n_counter,:] = np.zeros(number_synapses)
        store_time_difference[n_counter,:] += new_time_difference 

        # ****************************   
        # step 3: update the system
        # ****************************   

        # update time, number species, counter
        current_time += new_time_difference 
        current_species += np.transpose(stoch[next_r,:])
        n_counter += 1

        # store current system
        store_time[n_counter] = current_time
        store_number_molecules[n_counter,:] = current_species 
        store_filling_fraction_av += new_time_difference*current_species[:number_synapses].astype(np.float64)
        
        print("time: ", current_time, "n: ", n_counter)
        
    # prepare the final output
    store_time = store_time[:n_counter]
    store_number_molecules = store_number_molecules[:n_counter,:]
    store_time_difference = store_time_difference[:n_counter,:]
    
    # calculate final filling fraction
    filling_fraction = store_filling_fraction_av/(current_time*s_i)
    
    # calculate average of coefficient of variation
    mol_cv = store_number_molecules[:,:number_synapses] 
        # delete column for e_i and p since only w_i is relevant
    
    average = sum(store_time_difference*mol_cv)
    average /= current_time 
    
    coefficient_variation = np.sqrt(sum((mol_cv - average)**2*\
                (store_time_difference/current_time)))*100/average

    return(store_time, store_number_molecules, coefficient_variation, filling_fraction)

###################################################
######## Initialise the start values ##############
###################################################

# define the stochiometry of the substrates and products
reactions_alpha =\
"X8+X15->X1,X9+X15->X2,X10+X15->X3,X11+X15->X4,X12+X15->X5,X13+X15->X6,X14+X15->X7," 
reactions_beta =\
"X1->X8+X15,X2->X9+X15,X3->X10+X15,X4->X11+X15,X5->X12+X15,X6->X13+X15,X7->X14+X15,"
reactions_delta_gamma = "X15->0X15,0X15->X15"
all_reactions = reactions_alpha + reactions_beta + reactions_delta_gamma
sub_stoch, prod_stoch = \
reactions_stoch(all_reactions) 

# define the initial conditions

s_i = np.array([1,2,5,10,20,50,100]) # set number of slots
e_i = deepcopy(s_i)
w_i = deepcopy(s_i)
s = np.sum(s_i)
number_synapses = len(s_i)

F = 0.5 # set F for calculating alpha,gamma
phi = 2.67 # set phi for calculating alpha, gamma

beta = 60/43 # set beta
alpha = beta/(phi*s*(1-F)) # set alpha
delta = 1/14 # set delta
gamma = delta*(s*phi-(beta/alpha)) # set gamma

p = round(gamma/delta) # set p

tmax = 10 # set the end time 
n_max = 10000 # estimate n_max for later arrays
times_sim_av = 10 # number of repeated simulations for average

# prepare the rates 
rates = np.ones(len(s_i))*alpha
rates = np.append(rates,np.ones(len(s_i))*beta)
rates = np.append(rates,delta)
rates = np.append(rates,gamma)

# prepare initial number of molecules of each species according to filling fraction
w_i = np.round(w_i*F).astype(np.int64)
e_i -= w_i
init1 = np.append(np.append(w_i,e_i),p).astype(np.int64)

# check the result for filling fraction and CV
cv = np.zeros(number_synapses)
ff = np.zeros(number_synapses)

###################################################
############### Run the simulation ################
###################################################

counter_simulation = 0
while True:
    init = deepcopy(init1)
    if counter_simulation == times_sim_av:
        break
        
    # output the information how many simulations are already done
    counter_simulation += 1
    print("counter simulation: ", counter_simulation)
    
    results = gillespie_algo(s_i, init, rates, sub_stoch, prod_stoch, tmax, n_max)
    store_time = results[0]
    store_molecules = results[1]
    cv += results[2]
    ff += results[3]

    # txt files for number of molecules and time
    wi_data = open("w1_"+str(counter_simulation),"a+")
    for i in store_molecules:
        wi_data.write(str(list(i[:number_synapses]))+",\n")
    wi_data.close()
    
    ei_data = open("e1_"+str(counter_simulation),"a+")
    for i in store_molecules:
        ei_data.write(str(list(i[number_synapses:2*number_synapses]))+",\n")
    ei_data.close()
    
    p_data = open("p_"+str(counter_simulation),"a+")
    for i in store_molecules:
        p_data.write(str(i[-1])+",\n")
    p_data.close()
    
    time_data = open("time_"+str(counter_simulation),"a+")
    for i in store_time:
        time_data.write(str(i)+"\n")
    time_data.close()

# calculate the average of F and CV of all simulations
ff /= times_sim_av
cv /= times_sim_av

# store CV and F in txt file
cv_data = open("cv","a+")
for i in cv:
    cv_data.write(str(i)+"\n")
cv_data.close()

ff_data = open("F","a+")
for i in ff:
    ff_data.write(str(i)+"\n")
ff_data.close()

print("Average CV of w_i:", cv)
print("Average F of Synapse i:", ff)
print("The results are now saved in the txt files.")

###############################################################
# Plot w_i of each synapse during simulation: Fig. 4.3(a),(b) #
###############################################################

# extract the information from txt files
# data of w_i
sim = "1"
wi = "["
data = open("w1_"+str(sim))
for line in data:
    wi += line.rstrip()
wi = wi[:-1] 
wi += "]"

data.close()
wi = np.array(eval(wi))

# data of time
time = np.array([])
data = open("time"+"_"+sim)
for line in data:
    time = np.append(time, float(line.rstrip()))
data.close()

# plot
f = plt.figure(figsize=(4,3))
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
plt.rc('font', serif='Times New Roman') 
plt.rc('text', usetex=True)

color_lines = ['mediumturquoise','orchid','red','dimgray','blue', 'green','darkkhaki']
labels = [r'$s_1=1$', r'$s_2=2$', r'$s_3=5$',r'$s_4=10$', r'$s_5=20$', r'$s_6=50$',r'$s_7=100$']
for i in range(number_synapses-1,-1,-1):
    plt.plot(time, wi[:,i],'k',color=color_lines[i],label=labels[i],linewidth=0.3)
plt.xlabel(r'$t \; [{\rm min}]$', fontsize=12)
plt.ylabel(r'$w_i$', fontsize=12)
plt.legend(loc=1, fontsize=8)
plt.show()
f.savefig("plot_wi.pdf", bbox_inches='tight')

###################################################
######## Plot CV ##################################
###################################################

# this is a exemplary plot with one measurement of CV
# in file coefficientOfVariation.py you can find a plot that combines different CV's of multiple simulations into one figure

# extract the information from txt files
cv = np.array([])
data = open("cv")
counter = 0
for line in data:
    if counter == number_synapses:
        break
    cv = np.append(cv, float(line.rstrip())) 
    counter += 1
data.close()

# calculate regression
x_axis = s_i*F 
    # if you want to plot CV as a function of $w_i^\infty$ and F
    # if you want CV as a function of $s_i$:
    # x_axis = s_i

av_x_axis = sum(np.log(x_axis))/len(x_axis)

av_cv = sum(np.log(cv[:-1]))/len(cv[:-1])

b_cv=sum((np.log(x_axis)-av_x_axis)*(np.log(cv)-av_cv))/sum((np.log(x_axis)-av_x_axis)**2)
a_cv=av_cv-b_cv*av_x_axis

X = np.linspace(0.4,100,500,endpoint=True)
Y_cv = np.exp(a_cv)*(X**b_cv)

# plot
f = plt.figure(figsize=(4,3))
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
plt.rc('font', serif='Times New Roman') 
plt.rc('text', usetex=True)

plt.xscale('log')
plt.yscale('log')
plt.plot(x_axis, cv,'.',color='blue',label=r'$F=0.7$')
plt.plot(X, Y_cv,'-',color='blue',alpha = 0.3)
plt.ylabel(r'$\overline{VK_i} \; [{\rm \%}]$', fontsize=12) # in German it is Variationskoeffizient (VK)
plt.xlabel(r'$w_i^\infty$', fontsize=12) # if you want to plot CV as a function of $w_i^\infty$
    # if you want CV as a function of $s_i$:
    # plt.xlabel(r'$s_i$', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()
f.savefig("plot_one_cv.pdf", bbox_inches='tight')
