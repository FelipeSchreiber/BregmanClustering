import graph_tool.all as gt
import numpy as np

def get_SBM(num_nodes=6000,num_blocks=3,p_in=0.1,p_out=0.01,save=True,path="./"):
    sizes = ([ num_nodes // num_blocks ]*np.ones( num_blocks, dtype = int )).astype(int)
    block_membership = np.repeat(np.arange(num_blocks),sizes)
    graph = gt.Graph(directed=False)
    graph.add_vertex(num_nodes)

    # Assign block membership to nodes
    block_property = graph.new_vertex_property("int")
    for v, block_id in enumerate(block_membership):
        block_property[v] = block_id

    # Set edge probabilities based on block membership
    prob_matrix = np.full((num_blocks, num_blocks), p_out)
    np.fill_diagonal(prob_matrix, p_in)

    # Generate edges
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if np.random.rand() < prob_matrix[block_membership[u], block_membership[v]]:
                graph.add_edge(u, v)

    # Assign block membership as vertex property
    graph.vertex_properties["block_membership"] = block_property

    # Get the adjacency matrix
    adjacency_matrix = gt.adjacency(graph).toarray()

    if save == True:
        # Save the adjacency matrix as a numpy array
        np.save(path+f"graph_{num_nodes}_pin_{p_in}_pout_{p_out}.npy",\
                adjacency_matrix)
    
    return adjacency_matrix
