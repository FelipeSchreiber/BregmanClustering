import subprocess
import fileinput
import rpy2
import os
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr
from .cfg import *
if IGNORE_WARNINGS: 
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    logging.disable(logging.INFO)
    logging.basicConfig(level=logging.CRITICAL)


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
    #subprocess.call(["chmod","777",f"{bash_path}"])
    os.chmod(bash_path, 777)
    #os.environ["R_HOME"] = r"C:\\Program Files\R\R-4.2.3"
    subprocess.call([f"{bash_path}"])
    modify_csbm("./CSBM/Python/functions.py")
    modify_att_sbm("./AttributedSBM/FitAttribute.R")
    print("Installing R packages...\n This step takes about 5 min...\n")
    utils = importr("utils")
    packnames = ("igraph", "reticulate","mvtnorm")
    if CRAN_repo is None:
        utils.chooseCRANmirror(ind=1)
        utils.install_packages(StrVector(packnames))
    else:
        # Install packages
        utils.install_packages(StrVector(packnames),repos=CRAN_repo)
    other_algos_installed = True

if __name__ == "__main__":
    if not os.path.isdir("./sCSBM"):
        main()