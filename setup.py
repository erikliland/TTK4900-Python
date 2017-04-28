from setuptools import find_packages
from setuptools import setup

name = "pySimulator"
version = "0.1"
author = "Erik Liland"
author_email = "erik.liland@gmail.com"
description = "A framework for testing pyMHT"
license = "BSD"
keywords = 'simulation'
url = 'http://autosea.github.io/sf/2016/04/15/radar_ais/'
install_requires = ['matplotlib', 'numpy', 'scipy', 'termcolor']

packages = find_packages(exclude=['logs', 'data', 'profile'])
print("Packages", packages)

setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    description=description,
    license=license,
    keywords=keywords,
    packages=packages,
    include_package_data=True,
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'simAllSingle=pysimulator.runScenarios:mainSingle',
            'simAllMulti=pysimulator.runScenarios:mainMulti'
        ]
    }
)
