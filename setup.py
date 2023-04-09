import fnmatch
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as build_py_orig
from setuptools.command.install import install as _install
import os
from pkg_resources import resource_filename
import subprocess
import logging

class OverrideInstall(_install):

    def run(self):    
        mode = 777
        _install.run(self) # calling install.run(self) insures that everything that happened previously still happens, so the installation does not break! 
        # here we start with doing our overriding and private magic ..
        try:
            bash_path = resource_filename("BregmanTests","")+"/install_algos.sh"
            logging.log("Changing permissions of %s to %s" %
                        (bash_path, oct(mode)))
            os.chmod(bash_path, mode)
        except:
            logging.log("Failed to change permissions of install_algos.sh, please set manually to 777")
            pass
        
excluded = ['tests/*.ipynb','tests/*.eps','tests/.npy']

class build_py(build_py_orig):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            (pkg, mod, file)
            for (pkg, mod, file) in modules
            if not any(fnmatch.fnmatchcase(file, pat=pattern) for pattern in excluded)
        ]
    
setup(name='bregClust',
version='1.0',
description='A package for clustering attributed networks',
url='#',
author='Felipe Schreiber Fernandes, Maximilien Dreveton',
install_requires=["torch_geometric","rpy2==3.5.1","scikit-learn==1.2.2","numpy==1.24.2","scipy==1.10.1","matplotlib","joblib~=1.1.0"],
author_email='felipesc@cos.ufrj.br',
packages=find_packages(),
cmdclass={'build_py': build_py,"install":OverrideInstall},
package_data = {'BregmanTests': ['*.r',"*.sh"]},
zip_safe=False)