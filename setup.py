"""Uses setuptools to install the pytypeutils module"""
import setuptools
import os

setuptools.setup(
    name='pytypeutils',
    version='0.0.1',
    author='Timothy Moore',
    author_email='mtimothy984@gmail.com',
    description='Runtime typechecking without annotations',
    license='CC0',
    keywords='pytypeutils typechecking',
    url='https://github.com/tjstretchalot/pytypeutils',
    packages=['pytypeutils'],
    long_description=open(
        os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    long_description_content_type='text/markdown',
    install_requires=[],
    classifiers=(
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
        'Topic :: Utilities'),
    python_requires='>=3.6',
    extras_require={'test': ['numpy', 'torch']}
)
