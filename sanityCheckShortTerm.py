# stochastic simulation of AMPA receptor dynamics using the Gillespie algorithm
# Sanity Check: Plots different combinations of number of molecules and filling fraction of the system
# with all rates alpha, beta unlike 0 and delta, gamma = 0
# Figure 3.5(a), 3.6(a), 3.7(a) and tables 6.2, 6.3 of the manuscript

import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from copy import deepcopy
import time

###################################################
######## Functions ################################
###################################################

def FF(S,R,alpha,beta):
    """calculates theoretically short term filling fraction"""
    
    W=(S+R+(beta/alpha))/2-((S+R+(beta/alpha))**2/4-R*S)**(1/2)
    F=W/S
    return(F)

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
######## Initialise the values ####################
###################################################

# define the stochiometry of the substrates and products
sub_stoch, prod_stoch = \
reactions_stoch("X7+X4->X1,X7+X5->X2,X7+X6->X3,X1->X7+X4,X2->X7+X5,X3->X7+X6") 

# define the initial conditions

s_i = np.array([40,60,80]) # set number of slots
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
n_max = 5000 # estimate n_max for later arrays
times_sim_av = 1 # number of repeated simulations for average

# prepare the rates 
rates = np.ones(len(s_i))*alpha
rates = np.append(rates,np.ones(len(s_i))*beta)

# prepare initial number of molecules of each species according to filling fraction
w_i = np.round(w_i*F).astype(np.int64)
e_i -= w_i
init1 = np.append(np.append(w_i,e_i),p).astype(np.int64)

#####################################################
##### Check Filling Fraction: tables 6.2, 6.3 #######
############## and calculate CV #####################
#####################################################

cv = np.zeros(number_synapses)
ff = np.zeros(number_synapses)

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
print("Theoretical F:", FF(alpha,beta,delta,gamma))
print("The results are now saved in the txt files.")

###############################################################
######## Plot number of all substances: Fig. 3.5(a) ###########
###############################################################

# extract the information from txt files
# data of w_i
sim = "1"

substances = []
for i in ["w1_", "e1_", "p_"]:
    x = "["
    data = open(i+str(sim))
    for line in data:
        x += line.rstrip()
    x = x[:-1] 
    x += "]"
    data.close()
    substances.append(np.array(eval(x)))

wi, ei, p = substances[0], substances[1], substances[2]

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

plt.plot(time, p,'k',color='red',label=r'$p$',linewidth=0.3)

color_lines_e = ['cyan','green','blue']
color_lines_w = ['coral','black','darkviolet']
labels_e = [r'$e_1$', r'$e_2$', r'$e_3$']
labels_w = [r'$w_1$', r'$w_2$', r'$w_3$']
for i in range(number_synapses-1,-1,-1):
    plt.plot(time, ei[:,i],'k',color=color_lines_e[i],label=labels_e[i],linewidth=0.3)
    plt.plot(time, wi[:,i],'k',color=color_lines_w[i],label=labels_w[i],linewidth=0.3)
plt.xlabel(r'$t \; [{\rm min}]$', fontsize=12)
plt.ylabel(r'$Molekuelanzahl$', fontsize=12)
plt.legend(loc=1, fontsize=11)
plt.show()
f.savefig("long_all_substances.pdf", bbox_inches='tight')

###############################################################
### Plot number of all substances except p: Fig. 3.5(a) #######
###############################################################

f = plt.figure(figsize=(4,3))

for i in range(number_synapses-1,-1,-1):
    plt.plot(time, ei[:,i],'k',color=color_lines_e[i],label=labels_e[i],linewidth=0.3)
    plt.plot(time, wi[:,i],'k',color=color_lines_w[i],label=labels_w[i],linewidth=0.3)
plt.xlabel(r'$t \; [{\rm min}]$', fontsize=12)
plt.ylabel(r'$Molekuelanzahl$', fontsize=12)
plt.legend(loc=1, fontsize=11)
plt.show()
f.savefig("long_substances_without_p.pdf", bbox_inches='tight')

###############################################################
#### Plot number of slots and receptors: Fig. 3.5(a) ##########
###############################################################

f = plt.figure(figsize=(4,3))

R = np.sum(wi,1) + p
plt.plot(time, R,'k',color='deeppink',label=r'$R$',linewidth=0.3)

s_i = ei + wi
color_lines = ['coral','black','darkviolet']
labels = [r'$s_1$', r'$s_2$', r'$s_3$']
for i in range(number_synapses-1,-1,-1):
    plt.plot(time, s_i[:,i],'k',color=color_lines[i],label=labels[i],linewidth=0.3)
plt.xlabel(r'$t \; [{\rm min}]$', fontsize=12)
plt.ylabel(r'$Molekuelanzahl$', fontsize=12)
plt.legend(loc=1, fontsize=11)
plt.show()
f.savefig("long_slot_number_R.pdf", bbox_inches='tight')
