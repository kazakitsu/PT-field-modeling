from setuptools import setup, find_packages

setup(
    name="PT_field",
    version="0.2.0",
    description="Perturbative forward modeling at the field level",
    author="Kazuyuki Akitsu",
    url="https://github.com/kazakitsu/PT-field-modeling",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.1",
        "jax>=0.4.3",
        "jaxlib>=0.4.3",
        "lss_utils @ git+https://github.com/kazakitsu/lss_utils.git@main#egg=lss_utils",
    ],
    python_requires=">=3.10",
)
