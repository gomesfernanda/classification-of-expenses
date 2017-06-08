# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.me') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='basicsetup',
    version='0.1.0',
    description='basic setup for Classification of Expenses repo',
    long_description=readme,
    author='Fernanda Gomes',
    author_email='fergomes@gmail.com',
    url='https://github.com/gomesfernanda/classification-of-expenses',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)