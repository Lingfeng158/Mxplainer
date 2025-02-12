#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "generic.h"
#include "rule.h"

namespace py = pybind11;

PYBIND11_MODULE(fanCalcLib, m)
{
    m.def("formMinComb_c", &formMinCombination, R"pbdoc(
        Compute and form min combination in C++
    )pbdoc");
    m.def("calcFan_c", &calcFanWithMahJongGB, R"pbdoc(
        Compute Fan in C++
    )pbdoc");
    m.def("updateTileInfo_c", &updateTileInfo, R"pbdoc(
        update tile wall info in C++
    )pbdoc");
}