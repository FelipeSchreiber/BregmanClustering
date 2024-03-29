import subprocess
import fileinput
from sys import platform
import os
import stat
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr
from rpy2.rinterface_lib.embedded import RRuntimeError
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
    if not os.path.isdir('CSBM'):
        # st = os.stat(bash_path)
        # os.chmod(bash_path, st.st_mode | stat.S_IEXEC)
        os.chmod(bash_path, 0o777)
        print("Accessing "+bash_path)
        if platform == "win32":
            subprocess.call(["git","clone","https://github.com/glmbraun/CSBM/"])
            subprocess.call(["git","clone","https://github.com/stanleyn/AttributedSBM.git"])
            subprocess.call(["git","clone","https://github.com/MartijnGosgens/validation_indices"])
        else:
            subprocess.run(["sh",f"{bash_path}"])
        modify_csbm("./CSBM/Python/functions.py")
        modify_att_sbm("./AttributedSBM/FitAttribute.R")
    print("Installing R packages...\n This step takes about 5 min...\n")
    utils = importr("utils")
    packnames = ("igraph", "reticulate","mvtnorm")
    not_installed = []
    for pack in packnames:
            try:
                rpack = importr(pack)
            except Exception:
                not_installed.append(pack)
                pass
    packnames = tuple(not_installed)
    if CRAN_repo is None:
        utils.chooseCRANmirror(ind=1)
        utils.install_packages(StrVector(packnames))
    else:
        # Install packages
        utils.install_packages(StrVector(packnames),repos=CRAN_repo)
    other_algos_installed = True

if __name__ == "__main__":
    if not os.path.isdir("CSBM"):
        main()