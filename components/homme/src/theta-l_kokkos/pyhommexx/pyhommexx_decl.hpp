#ifndef PYHOMMEXX_DECL_HPP
#define PYHOMMEXX_DECL_HPP

#include "Hommexx_config.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace Homme {
namespace pyhommexx {

namespace nb = nanobind;

template<int N>
using ndarray_t = nb::ndarray<double,nb::ndim<N>>;

void init (nb::object py_comm, const nb::str& nml_filename);
nb::dict get_params();

void get_num_unique_pts (nb::ndarray<int>& n);
void get_unique_pts (nb::ndarray<int>& ia,
                     nb::ndarray<int>& ja);

void set_state (const nb::ndarray<double>& uv,
                const nb::ndarray<double>& vthdp,
                const nb::ndarray<double>& dp);

void get_state (nb::ndarray<double>& uv,
                nb::ndarray<double>& vthdp,
                nb::ndarray<double>& dp);

void forward(const double dt);

void finalize();

} // namespace pyhommexx
} // namespace Homme

#endif // PYHOMMEXX_UTILS_HPP
