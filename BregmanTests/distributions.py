import numpy as np

## Given a specified tuple of desired mean and variance, obtain scale and shape for numpy.random.gamma
def get_gamma_parameter(mean,variance):
    scale = variance/mean
    shape = mean/scale
    return (scale,shape)

def get_normal_parameter(mean,variance):
    scale = np.sqrt(variance)
    return (mean,scale)
    
def get_wald_parameter(mean,variance):
	return (mean,np.power(mean,3)/variance)

def get_logistic_parameter(mean,variance):
	return (mean,np.sqrt(3*variance/np.power(np.pi,2)))

def get_exponential_parameter(mean,variance):
    scale = 1/mean
    return (scale,)

def get_poisson_parameter(mean,variance):
    return (mean,)

def make_weight_params(f):
	def get_parameters_for_every_community_pair(means,variance,n_clusters):
		params = {}
		for i in range(n_clusters):
			params[i] = {}
			for j in range(n_clusters):
				params[i][j] = f(means[i,j],variance)
		return params
	return get_parameters_for_every_community_pair

distributions_dict = {
"gamma":(np.random.gamma,get_gamma_parameter),
"gaussian":(np.random.normal,get_normal_parameter),
"wald":(np.random.wald,get_wald_parameter),
"logistic":(np.random.logistic,get_logistic_parameter),
"exponential":(np.random.exponential,get_exponential_parameter),
"poisson":(np.random.poisson,get_poisson_parameter)
}
