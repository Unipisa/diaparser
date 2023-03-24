# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='diaparser',
    version='1.1.3',
    author='Yu Zhang, Giuseppe Attardi',
    author_email='yzhang.cs@outlook.com, attardi@di.unipi.it',
    description='Direct Attentive Dependency Parser',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Unipisa/diaparser',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic'
    ],
    setup_requires=[
        'setuptools>=18.0',
    ],
    # stanza 1.3 has incompatible changes (attribute feat_dropout instead of dropout)
    install_requires=['torch>=2.0', 'transformers', 'nltk', 'stanza', 'numpy'],
    entry_points={
        'console_scripts': [
            'diaparser=diaparser.cmds.biaffine_dependency:main',
        ]
    },
    python_requires='>=3.6',
    zip_safe=False
)
