import numpy as np

def get_gamma_parameters(means,variance,n_clusters):
    params = {}
    curr = 0
    for i in range(n_clusters):
        params[i] = {}
        for j in range(i,n_clusters):
            scale = variance/means[curr]
            shape = means[curr]/scale
            params[i][j] = (shape,scale)
            curr += 1
    return params

def get_normal_parameters(means,variance,n_clusters):
    params = {}
    curr = 0
    scale = np.sqrt(variance)
    for i in range(n_clusters):
        params[i] = {}
        for j in range(i,n_clusters):
            params[i][j] = (means[i],scale)
            curr += 1
    return params

distributions_dict = {
"gamma":(np.random.gamma,get_gamma_parameters),
"gaussian":(np.random.normal,get_normal_parameters)
}
