#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name="gym_cog_ml_tasks",
    version="0.0.1",
    install_requires=["gym>=0.17.0", "numpy>=1.18.1"],
    packages=find_packages(),
)