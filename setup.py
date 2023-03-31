import setuptools
setuptools.setup(name='bregClust',
version='0.1',
description='A package for clustering attributed networks',
url='#',
author='Felipe Schreiber Fernandes',
install_requires=["torch_geometric","torch","scikit-learn>=1.2.2","numpy>=1.22.4","scipy>=1.10.1"],
author_email='felipesc@cos.ufrj.br',
packages=setuptools.find_packages(exclude=["tests"]),
zip_safe=False)
