'''
# Incomplete: I was looking at network visualizations. Not much emerged.
'''


user_pairings = []
for thread_id, clique in emails.groupby('thread_id').sender:
    for user_A, user_B in zip(clique[1:], clique[:1]):
    #for user_A, user_B in combinations(clique, 2):
        if 'log in to unmask' in (user_A, user_B):
            continue
        pairing = tuple(sorted([user_A, user_B]))
        user_pairings.append(pairing)
edge_list = pd.DataFrame(user_pairings, columns=['A', 'B'])
edge_list = edge_list.groupby(['A', 'B']).size().reset_index()
user_graph = nx.from_pandas_dataframe(edge_list, 'A', 'B', 0)
nx.draw_networkx(user_graph)
plt.axis("off")
ax = plt.gca() # to get the current axis
ax.collections[0].set_edgecolor('none') 
plt.show()

from pyvis.network import Network
G = Network(notebook=True)
G.from_nx(user_graph)
G.show('results/graph.html')