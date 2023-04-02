from pkg_resources import resource_filename
path_to_data = "./data/Benchmark/"
path_to_att_sbm = "./AttributedSBM/FitAttribute.R"
base_path = resource_filename("BregmanTests","")
bash_path = base_path+"/install_algos.sh"
CRAN_repo = None