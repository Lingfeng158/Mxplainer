### Goal of src_c

The goal of src_c is to port python version of the program into C++ version, in order to speed up calculation time.

### Modified Design

Python Format: 
```
handwall_dict: {"Tile A":1, "shown":[{"Tile B":1, "AnGang":True}]}
```
C++ Format breaks down into two variables:
```
handwall_dict: {"Tile A":1}
pack_list: [{"Tile B":1, "AnGang":1},{...},{...}]
```

### Generating CPython Extention from C++

Pre-requisite for compilation: G++/gcc, c++11, python3.6.5(for botzone)/python3.x(for all other environments), pybind11

Hard requirement for botzone: G++/gcc version <= 9, i.e. `sudo apt-get install g++-8 gcc-8` for installing g++/gcc 8.4.0

1. Locate directory for Python.h, i.e. `/home/billy/anaconda3/envs/py36/include/python3.6m`
2. Locate directory for pybind11, i.e. `/home/billy/anaconda3/envs/py36/include/`, which can be found through `import sys`, `sys.path`
3. Replace pyModule.cpp's `PYBIND11_MODULE(fanCalcLib, m)` with newly defined shared-library name, if applicable. 
4. Compile shared-object with name "fanCalcLib.so", with command: `g++-8 -O4 -shared -fPIC -I<path_to_python> -I<path_to_pybind11> -o fanCalcLib.so <path_to_src_c>/!(driver).cpp`
5. Move "fanCalcLib.so" into src/ folder.

PS1. Example of default anaconda virtual environment installation: ```g++-8 -O3 -shared -fPIC -I/home/billy/anaconda3/include/python3.9 -I/home/billy/anaconda3/lib/python3.9/site-packages/pybind11/include -o fanCalcLib.so src_c/!(driver).cpp```

PS2. Example of additional anaconda virtual environment installation: ```g++-8 -O3 -shared -fPIC -I/home/billy/anaconda3/envs/py36/include/python3.6m -I/home/billy/anaconda3/envs/py36/include/ -o fanCalcLib.so src_c/!(driver).cpp```

### Uploading AI to botzone

1. upload .so file to botzone, under /data folder.
2. upload other necessary files excluding .so file as python file to botzone.