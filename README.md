# Mxplainer
An analytical tool for Official International MahJong game.

# Directory Introduction

## src/
Survey bot and human players' characteristics exhibited in historical gameplays.

## data_src/
compressed datasets.

## data/
Under which are processed MahJong match data, ready to use.

## approximation/
Code for convert historical data into a format suitable for supervised learning, neural version of framework(as models), and scripts for training parameters.

## bot_framework_slim/
Code for uploading fitted bots to botzone.org.cn

## search_component_py/
Search component code in python

## search_component_c/
Search component code in C++ for more efficient computation. 
C++ code has been compiled to fanCalcLib.so for Python 3.6 and FanCalcLibPy39.so for Python 3.9
