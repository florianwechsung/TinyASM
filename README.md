# TinyASM

[![<florianwechsung>](https://circleci.com/gh/florianwechsung/TinyASM.svg?style=svg)](https://app.circleci.com/pipelines/github/florianwechsung/TinyASM)


A simple implementation of PETSc's ASM preconditioner that is focussed on the
case of small matrices. We avoid the overhead of KSP and PC objects for each
block and just use the dense inverse.

## Installation

You need a recent firedrake install

    curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
    python3 firedrake-install

Then there are two variants:

A) Simply run

    pip install git+https://github.com/florianwechsung/TinyASM

B) clone the repo including submodules:

    git clone --recursive git@github.com:florianwechsung/TinyASM.git

and then
    
    cd TinyASM
    pip3 install -e .

