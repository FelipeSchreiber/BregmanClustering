import subprocess
import fileinput
import cfg
import rpy2
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr
import os

path = "./data/Benchmark/"
path_to_att_sbm = "./AttributedSBM/FitAttribute.R"

def modify_att_sbm(path):
    for i,line in enumerate(fileinput.input(path, inplace=True)):
        if i == 0: 
            print(f'{line.replace(")",",kmeansinit)")}', end='')
        elif i ==  27:
            print(f'#{line}', end='')
        else:
            print(line)

def modify_csbm(path):
    for i,line in enumerate(fileinput.input(path, inplace=True)):
        if "306" in line: 
            print(f'{line.replace("306","n")}')
        else:
            print(line)

def main():
    print("Downloading packages from github...\n")
    #bash_path = os.path.abspath(bregClust.__file__)
    bash_path = f"{cfg.install_breg_path}/BregmanTests"
    print(bash_path)
    subprocess.call([f"{bash_path}/install_algos.sh"])
    modify_csbm("./CSBM/Python/functions.py")
    modify_att_sbm(path_to_att_sbm)
    print("Installing R packages...\n")
    utils = importr('utils')
    utils.chooseCRANmirror(ind=1)
    # Install packages
    packnames = ("igraph", "reticulate","mvtnorm")
    utils.install_packages(StrVector(packnames))

if __name__ == "__main__":
    main()