from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='DANTE',
      version='0.1',
      description="Package for Stochastic Point Process Information Extractor.\
                    Contains all the base classes for definining any point process along with accompanying features",
      long_description=long_description,
      author="Akshay Aravamudan",
      author_email="akshay.aravamudan@gmail.com",
      packages=['',
                'core',
                'utils',
                'point_processes'],
      url="https://github.fit.edu/Information-Diffusion-Research-Group/split-population-survival-exploits.git",
      package_dir={'': 'splitPopulationSurvivalExploits/src'},
      python_requires='>=3.6')
