import fnmatch
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as build_py_orig

excluded = ['tests/*.ipynb','tests/*.eps','tests/.npy']

class build_py(build_py_orig):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            (pkg, mod, file)
            for (pkg, mod, file) in modules
            if not any(fnmatch.fnmatchcase(file, pat=pattern) for pattern in excluded)
        ]
    
s = setup(name='bregClust',
version='0.1',
description='A package for clustering attributed networks',
url='#',
author='Felipe Schreiber Fernandes, Maximilien Dreveton',
install_requires=["torch_geometric","rpy2"],
author_email='felipesc@cos.ufrj.br',
packages=find_packages(),
cmdclass={'build_py': build_py},
package_data = {'tests': ['*.r']},
zip_safe=False)
global install_breg_path
install_breg_path = s.command_obj['install'].__dir__()