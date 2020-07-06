from pylab import *
import networkx as nx
import pandas as pd
import itertools as itr
from scipy import spatial
from collections import Counter
import operator
from collections import OrderedDict


dim = 4  # dim: dimensions of problem space
group_size = 20
number_items = 10
n = 3  # number of ideas in final selection
MC = 10000  # number of Monte Carlo experiments
mont_iter = 20  # number of Monte Carlo experiments for each group

coef_unknown = 0.2
coef_com_like = 1.
number_new_ideas = 4  # number of potential new ideas in function of post a novel idea
number_initial_idea = 100
noise = 0.1
number_personal_idea = 1
number_visible_idea = 20  # number of visible ideas in each iteration
number_groups = 1


########################################################################################################################
# functions
########################################################################################################################
items = list(arange(number_items))
initial_n_idea = [[choice(items) for i in range(dim)] for j in range(number_initial_idea)]
TU = {tuple(c): random() for c in initial_n_idea}
i_1, i_0 = choice(arange(number_initial_idea), 2, replace=False)
TU[tuple(initial_n_idea[i_1])] = 1.
TU[tuple(initial_n_idea[i_0])] = 0.

def True_utility(idea): # given n initial ideas
    pure_idea = tuple(idea['idea'])
    initial_idea = list(TU.keys())
    initial_idea_value = list(TU.values())
    if pure_idea in initial_idea:
        return TU[pure_idea]
    else:
        dist = []
        for i in range(len(initial_idea)): # n is the number of initial ideas
            dist.append(sqrt(sum([(float(initial_idea[i][d]) - float(pure_idea[d]))**2 for d in range(dim)])))
        return sum([initial_idea_value[i]*(dist[i]**(-2)) for i in range(len(dist))])/sum([dist[i]**(-2) for i in range(len(dist))])


def Background(expertise):
    Background = []
    for i in range(dim):
        if i in expertise:
            Background.append(list(choice(items, int(uniform(int(number_items/2.)+2, number_items)), replace=False)))
        else:
            Background.append(list(choice(items, int(uniform(1, int(number_items/2.)-2)), replace=False)))
    return Background


def Diff_background(agent1, agent2):
    Background1 = agent1.background
    Background2 = agent2.background
    overlap = 0
    for i in range(dim):
        for j in Background1[i]:
            if j in Background2[i]:
                overlap += 1
    whole_items1 = sum([len(Background1[i]) for i in range(dim)])
    whole_items2 = sum([len(Background2[i]) for i in range(dim)])
    return 1-overlap/(whole_items1 + whole_items2 - overlap)


def Individual_utility(agent, idea):
    background = agent.background
    pure_idea = idea['idea']
    r = 0
    for i in range(dim):
        if pure_idea[i] in background[i]:
            r += 1
    return r*clip(True_utility(idea)+uniform(-noise, noise), 0, 1) + coef_unknown*(1-r)*uniform(0, 1.)


def Best_idea(agent):
    personal_idea_pool = agent.idea_pool
    if len(personal_idea_pool) == 0:
        return None
    else:
        max_idea = max(personal_idea_pool, key=lambda i: Individual_utility(agent, i))
    return max_idea


def Variance_ideas(idea_list):
    dist = []
    for i in idea_list:
        for j in idea_list:
            if i != j:
                dist.append(sqrt(sum([(float(i[d])-float(j[d]))**2 for d in range(dim)])))
    return var(dist)


def Probability_accept(node, idea):
    agent = node['agent']
    best_utility = max([Individual_utility(node['agent'], i) for i in agent.idea_pool])
    mid_utility = median([Individual_utility(node['agent'], i) for i in agent.idea_pool])
    if Individual_utility(node['agent'], idea) > best_utility:
        return 1.
    elif Individual_utility(node['agent'], idea) < mid_utility:
        return 0.
    else:
        if mid_utility == best_utility:
            return 1.
        else:
            return (Individual_utility(node['agent'], idea)-mid_utility)/(best_utility-mid_utility)


def Visible_ideas(node):  # personal ideas are not included
    local_ideas = []
    # get the index of the node
    index = 0
    for i in range(len(net.nodes)):
        if node == net.nodes[i]:
            index = i
    for nb in list(net.neighbors(index)):  # neighbors' idea_pool
        local_ideas = local_ideas + net.nodes[nb]['agent'].idea_pool

    local_iter = [i['iteration'] + 10**-10 for i in local_ideas]
    local_comment = [len(i['comment']) + 10**-10 for i in local_ideas]
    local_like = [len(i['like']) + 10**-10 for i in local_ideas]

    p_iter = [(i-min(local_iter))/max(local_iter) for i in local_iter]
    p_comment = [(i-min(local_comment))/max(local_comment) for i in local_comment]
    p_like = [(i-min(local_like))/max(local_like) for i in local_like]
    p = [p_iter[i]+coef_com_like*p_comment[i]+coef_com_like*p_like[i] for i in range(len(p_iter))]
    if sum(p) == 0:
        norm_p = [1./len(p)]*len(p)
    else:
        norm_p = [i/sum(p) for i in p]

    if len(local_ideas) <= number_visible_idea:
        return local_ideas
    else:
        return list(choice(local_ideas, number_visible_idea, p=norm_p))



def Like(node, idea):  # need to be testified
    if node in idea['like']:
        return
    elif random() <= Probability_accept(node, idea):
        idea['like'].append(node)
        print("like")
    else:
        return


def Comment(node, idea): # comment here represents the suggestions; complement comments are assumed to be equal to clicking like
    agent = node['agent']
    pure_idea = idea['idea']
    personal_pure_ideas = [i['idea'] for i in agent.idea_pool]

    if node in idea['comment']:
        return
    elif random() <= Probability_accept(node, idea):
        if pure_idea in personal_pure_ideas:  # if the agent has the same idea
            idea['comment'].append(node)
        else:
            new_idea = pure_idea[:]
            background = agent.background
            size_background = sum([len(i) for i in background])
            selected_dim = choice(list(arange(dim)), p=[(len(i)+10**-10)/size_background for i in background])
            random_item = choice(background[selected_dim])
            new_idea[selected_dim] = random_item
            new_idea_dic = {'idea': new_idea}
            if Individual_utility(agent, new_idea_dic) > Individual_utility(agent, idea):  # if the new idea is better than previous one
                idea['comment'].append(node)
            print("comment")
    else:
        return


def Post_novel_idea(node, iteration):
    agent = node['agent']
    background = agent.background
    potential_new_ideas = []
    for i in range(number_new_ideas):
        tem_new_idea = []
        for di in range(dim):  # choose item randomly
            tem_new_idea.append(choice(background[di]))
        potential_new_ideas.append({'idea': tem_new_idea})

    selected_idea = potential_new_ideas[0]
    for i in potential_new_ideas:
        if Individual_utility(agent, i) > Individual_utility(agent, selected_idea):
            selected_idea = i
    new_idea = {'idea': selected_idea['idea'], 'comment': [], 'like': [], 'iteration': iteration}

    personal_idea_pool = node['agent'].idea_pool
    pure_personal_idea = [i['idea'] for i in personal_idea_pool]
    pure_visible_idea = [i['idea'] for i in visible_ideas]

    if (new_idea['idea'] not in pure_personal_idea) and (new_idea['idea'] not in pure_visible_idea)\
            and (random() <= Probability_accept(node, new_idea)):
        node['agent'].idea_pool.append(new_idea)
        print("new")
    else:
        return

def Post_revised_idea(node, iteration):
    agent = node['agent']
    personal_idea_pool = node['agent'].idea_pool
    pure_personal_ideas = [i['idea'] for i in personal_idea_pool]
    potential_ideas = []
    for i in visible_ideas:
        if i['idea'] not in pure_personal_ideas:
            potential_ideas.append(i)

    if len(potential_ideas) != 0:  # revise an idea that is randomly selected
        selected_idea = potential_ideas[choice(len(potential_ideas))]
        selected_pure_idea = selected_idea['idea']

        potential_new_ideas = []
        for i in range(number_new_ideas):
            tem_new_idea = selected_pure_idea[:]
            background = agent.background
            size_background = sum([len(i) for i in background])
            selected_dim = choice(list(arange(dim)), p=[(len(i) + 10 ** -10) / size_background for i in background])
            modified_item = Best_idea(agent)['idea'][selected_dim]
            tem_new_idea[selected_dim] = modified_item
            tem_new_idea_dic = {'idea': tem_new_idea}
            potential_new_ideas.append(tem_new_idea_dic)

        selected_idea = potential_new_ideas[0]
        for i in potential_new_ideas:
            if Individual_utility(agent, i) > Individual_utility(agent, selected_idea):
                selected_idea = i
        new_idea = {'idea': selected_idea['idea'], 'comment': [], 'like': [], 'iteration': iteration}

        personal_idea_pool = node['agent'].idea_pool
        pure_personal_idea = [i['idea'] for i in personal_idea_pool]
        pure_visible_idea = [i['idea'] for i in visible_ideas]
        pure_new_idea = new_idea['idea']

        if (pure_new_idea not in pure_personal_idea) and (pure_new_idea not in pure_visible_idea):
            #and (random() <= Probability_accept(node, new_idea)):
            node['agent'].idea_pool.append(new_idea)
            print("revised")
        else:
            return
    else:
        return


def Post_existed_idea(node, iteration):
    personal_idea = node['agent'].idea_pool
    pure_personal_idea = [i['idea'] for i in personal_idea]
    qualified_ideas = []
    for i in visible_ideas:
        if (i['idea'] not in pure_personal_idea) and (Individual_utility(node['agent'], i) >= Individual_utility(node['agent'], Best_idea(node['agent']))):
            qualified_ideas.append(i)

    if len(qualified_ideas) == 0:
        return
    else:
        popular_qualified_ideas = [len(i['like']) + len(i['comment']) + 10**-10 for i in qualified_ideas]
        p = [i/sum(popular_qualified_ideas) for i in popular_qualified_ideas]
        tem_idea = choice(qualified_ideas, p=p)
        new_idea = tem_idea.copy()
        new_idea['iteration'] = iteration
        node['agent'].idea_pool.append(new_idea)
        print("copy")


def Generate_three_group(agent_list):
    whole_local_diversity = {}
    tem_list = agent_list[:]
    for mont in range(MC):
        shuffle(tem_list)
        for i in range(group_size):
            net.nodes[i]['agent'] = tem_list[i]

        local_diversity = []
        for i in net.nodes:
            tem_diff = []
            for nei in list(net.neighbors(i)):
                tem_diff.append(Diff_background(net.nodes[i]['agent'], net.nodes[nei]['agent']))
            local_diversity.append(mean(tem_diff))
        whole_local_diversity[tuple(tem_list)] = mean(local_diversity)

    sorted_whole_local_diversity = OrderedDict(sorted(whole_local_diversity.items(), key=lambda x: x[1]))

    Low = list(sorted_whole_local_diversity.keys())[0]
    Medium = list(sorted_whole_local_diversity.keys())[int(MC/2.)]
    Large = list(sorted_whole_local_diversity.keys())[-1]
    Low_value = list(sorted_whole_local_diversity.values())[0]
    Medium_value = list(sorted_whole_local_diversity.values())[int(MC/2.)]
    Large_value = list(sorted_whole_local_diversity.values())[-1]
    return Low, Medium, Large, Low_value, Medium_value, Large_value


def Final_selection(node, n):
    personal_idea_pool = node['agent'].idea_pool
    whole_ideas = personal_idea_pool + visible_ideas
    sorted_ideas = sorted(whole_ideas, key=lambda i: Individual_utility(node['agent'], i))
    return sorted_ideas[-n:]

def All_ideas(net):
    all_ideas = []
    for nod in net.nodes:
        all_ideas = all_ideas + [j['idea'] for j in net.nodes[nod]['agent'].idea_pool]
    return all_ideas




########################################################################################################################
# initialization
########################################################################################################################
background_list1 = [Background([0,1])  for j in range(int(group_size/2.))]
background_list2 = [Background([2,3])  for j in range(int(group_size/2.))]
background_list = background_list1 + background_list2
participation_tendency = [1.]*group_size

###  creat agent
class agent:
    pass
# create a small world network
net = nx.random_regular_graph(d=4, n=20)

# initialization
agent_list = []
for i in range(group_size):
    ag = agent()
    ag.index = i
    ag.participation_tendency = participation_tendency[i]
    ag.background = background_list[i]
    ag.idea_pool = []
    # add agent into agent list
    agent_list.append(ag)

########################################################################################################################
# generate three groups: cluster, random, and dispersed
########################################################################################################################
#return Low, Medium, Large, Low_value, Medium_value, Large_value
Low, Medium, Large, Low_value, Medium_value, Large_value = Generate_three_group(agent_list)
print('Low value', Low_value)
print('Medium value', Medium_value)
print('Large value', Large_value)



########################################################################################################################
# iteration
########################################################################################################################
number_iteration = 10
cluster_utility = []
cluster_ideas = []
for c in range(mont_iter):
    for i in range(group_size):
        Low[i].idea_pool = [{'idea': [choice(ag.background[d]) for d in range(dim)], 'comment': [], 'like': [], 'iteration': 0} for n in range(number_personal_idea)]
        net.nodes[i]['agent'] = Low[i]

    final_ideas = []
    for it in range(number_iteration):
        iter = it + 1
        print(iter)
        for nod in net.nodes:
            print('node: ', nod)
            if random() < net.nodes[nod]['agent'].participation_tendency:
                # agent behaviors
                visible_ideas = Visible_ideas(net.nodes[nod])
                # like
                for vi in visible_ideas:
                    Like(net.nodes[nod], vi)
                # comment. to be finished
                for vi in visible_ideas:
                    Comment(net.nodes[nod], vi)
                # post novel ideas
                Post_novel_idea(net.nodes[nod], iter)
                # post revised ideas
                Post_revised_idea(net.nodes[nod], iter)
                # advocacy
                Post_existed_idea(net.nodes[nod], iter)

    whole_ideas = []
    whole_ideas.append(All_ideas(net))

    for i in net.nodes:
        final_ideas = final_ideas + Final_selection(net.nodes[i], n)

    final_pure_ideas = [tuple(i['idea']) for i in final_ideas]
    most_supported_idea = {'idea':list(Counter(final_pure_ideas).most_common()[0][0])}
    cluster_utility.append(True_utility(most_supported_idea))

number_cluster = [len(i) for i in whole_ideas]
variance_cluster = [Variance_ideas(i) for i in whole_ideas]

############################################################

random_utility = []
random_ideas = []
for r in range(mont_iter):
    for i in range(group_size):
        Medium[i].idea_pool = [{'idea': [choice(ag.background[d]) for d in range(dim)], 'comment': [], 'like': [], 'iteration': 0} for n in range(number_personal_idea)]
        net.nodes[i]['agent'] = Medium[i]

    final_ideas = []
    for it in range(number_iteration):
        iter = it + 1
        for nod in net.nodes:
            if random() < net.nodes[nod]['agent'].participation_tendency:
                # agent behaviors
                visible_ideas = Visible_ideas(net.nodes[nod])
                # like
                for vi in visible_ideas:
                    Like(net.nodes[nod], vi)
                # comment. to be finished
                for vi in visible_ideas:
                    Comment(net.nodes[nod], vi)
                # post novel ideas
                Post_novel_idea(net.nodes[nod], iter)
                # post revised ideas
                Post_revised_idea(net.nodes[nod], iter)
                # advocacy
                Post_existed_idea(net.nodes[nod], iter)

    whole_ideas = []
    whole_ideas.append(All_ideas(net))
    for i in net.nodes:
        final_ideas = final_ideas + Final_selection(net.nodes[i], n)

    final_pure_ideas = [tuple(i['idea']) for i in final_ideas]
    most_supported_idea = {'idea':list(Counter(final_pure_ideas).most_common()[0][0])}
    random_utility.append(True_utility(most_supported_idea))

number_random = [len(i) for i in whole_ideas]
variance_random = [Variance_ideas(i) for i in whole_ideas]



dispersed_utility = []
dispersed_ideas = []
for l in range(mont_iter):
    for i in range(group_size):
        Large[i].idea_pool = [{'idea': [choice(ag.background[d]) for d in range(dim)], 'comment': [], 'like': [], 'iteration': 0} for n in range(number_personal_idea)]
        net.nodes[i]['agent'] = Large[i]

    final_ideas = []
    for it in range(number_iteration):
        iter =  it + 1
        for nod in net.nodes:
            if random() < net.nodes[nod]['agent'].participation_tendency:
                # agent behaviors
                visible_ideas = Visible_ideas(net.nodes[nod])
                # like
                for vi in visible_ideas:
                    Like(net.nodes[nod], vi)
                # comment. to be finished
                for vi in visible_ideas:
                    Comment(net.nodes[nod], vi)
                # post novel ideas
                Post_novel_idea(net.nodes[nod], iter)
                # post revised ideas
                Post_revised_idea(net.nodes[nod], iter)
                # advocacy
                Post_existed_idea(net.nodes[nod], iter)

    whole_ideas = []
    whole_ideas.append(All_ideas(net))
    for i in net.nodes:
        final_ideas = final_ideas + Final_selection(net.nodes[i], n)

    final_pure_ideas = [tuple(i['idea']) for i in final_ideas]
    most_supported_idea = {'idea':list(Counter(final_pure_ideas).most_common()[0][0])}
    dispersed_utility.append(True_utility(most_supported_idea))

number_dispersed = [len(i) for i in whole_ideas]
variance_dispersed = [Variance_ideas(i) for i in whole_ideas]



print('Clustered diversity: ', cluster_utility)
print('Random diversity: ', random_utility)
print('Dispersed: ', dispersed_utility)


df = pd.DataFrame(list(zip(cluster_utility, random_utility, dispersed_utility)), columns =['Clustered', 'Randomly Distributed', 'Dispersed'])
ax = plt.subplots(figsize=(8, 4))
df.boxplot(grid=False)
plt.ylabel('Utility of the most supported idea')
plt.grid(axis='y', linestyle=':', color='lightgrey')
#plt.ylim(0.0, 1.0)
plt.show()


df = pd.DataFrame(list(zip(number_cluster, number_random, number_dispersed)), columns =['Clustered', 'Randomly Distributed', 'Dispersed'])
ax = plt.subplots(figsize=(8, 4))
df.boxplot(grid=False)
plt.ylabel('Number of distinct ideas')
plt.grid(axis='y', linestyle=':', color='lightgrey')
plt.show()


df = pd.DataFrame(list(zip(variance_cluster, variance_random, variance_dispersed)), columns =['Clustered', 'Randomly Distributed', 'Dispersed'])
ax = plt.subplots(figsize=(8, 4))
df.boxplot(grid=False)
plt.ylabel('Variance of ideas')
plt.grid(axis='y', linestyle=':', color='lightgrey')
plt.show()

