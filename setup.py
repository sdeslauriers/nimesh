from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='nimesh',
    version='0.0.0',
    packages=['nimesh', 'nimesh.io'],
    url='https://github.com/sdeslauriers/nimesh',
    license='MIT',
    author='Samuel Deslauriers-Gauthier',
    author_email='sam.deslauriers@gmail.com',
    description='A Python package to manipulate neuroimaging triangular '
                'meshes.',
    long_description=long_description,
    long_description_content_type='test/markdown',
    install_requires=['numpy', 'nibabel'],
    extras_requires={
        'visualization': ['vtk'],
    },
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.'
    ),
)
