from setuptools import setup

setup(
    name='nimesh',
    version='0.0.0',
    packages=['nimesh'],
    url='https://github.com/sdeslauriers/nimesh.git',
    license='MIT',
    author='Samuel Deslauriers-Gauthier',
    author_email='sam.deslauriers@gmail.com',
    description='A Python package to manipulate neuroimaging triangular '
                'meshes.',
    install_requires=['numpy', 'nibabel']
)
