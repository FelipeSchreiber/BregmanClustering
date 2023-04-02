import subprocess
import fileinput
import rpy2
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr
from cfg import *

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
    subprocess.call(["chmod","777",f"{bash_path}"])
    subprocess.call([f"{bash_path}"])
    modify_csbm("./CSBM/Python/functions.py")
    modify_att_sbm("./AttributedSBM/FitAttribute.R")
    print("Installing R packages...\n")
    utils = importr('utils')
    #utils.chooseCRANmirror(ind=1)
    # Install packages
    packnames = ("igraph", "reticulate","mvtnorm")
    utils.install_packages(StrVector(packnames),repos="https://cran.fiocruz.br/")

if __name__ == "__main__":
    main()