import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='transformer-utils',
    description="Large autoregressive language modeling helpers",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/nostalgebraist/transformer-utils",
    author="nostalgebraist",
    author_email="nostalgebraist@gmail.com",
    license="MIT",
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'transformers',
        'seaborn',
        'tqdm'
    ],
)
