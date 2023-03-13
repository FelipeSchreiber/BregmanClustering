import numpy as np

def get_gamma_parameter(mean,variance):
    scale = variance/mean
    shape = mean/scale
    return (scale,shape)

def get_gamma_parameters(means,variance,n_clusters):
    params = {}
    curr = 0
    for i in range(n_clusters):
        params[i] = {}
        for j in range(i,n_clusters):
            params[i][j] = get_gamma_parameter(means[curr],variance)
            curr += 1
    return params

def get_normal_parameter(mean,variance):
    scale = np.sqrt(variance)
    return (mean,scale)

def get_normal_parameters(means,variance,n_clusters):
    params = {}
    curr = 0
    for i in range(n_clusters):
        params[i] = {}
        for j in range(i,n_clusters):
            params[i][j] = get_normal_parameter(means[i],variance)
            curr += 1
    return params

distributions_dict = {
"gamma":(np.random.gamma,get_gamma_parameters,get_gamma_parameter),
"gaussian":(np.random.normal,get_normal_parameters,get_normal_parameter)
}
