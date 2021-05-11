from setuptools import setup, find_packages

setup(
    name='transformer-utils',
    version='0.0.3',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'transformers',
        'pandas~=0.24.2',
        'seaborn',
        'tqdm'
    ],
)
