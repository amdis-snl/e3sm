#ifndef PYHOMMEXX_DECL_HPP
#define PYHOMMEXX_DECL_HPP

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace pyhommexx {

namespace nb = nanobind;

template<int N>
using ndarray_t = nb::ndarray<double,nb::ndim<N>>;

void init (nb::object py_comm, const nb::str& nml_filename);
nb::dict get_params();

void get_num_unique_pts (nb::ndarray<int>& n);
void get_unique_pts (nb::ndarray<int>& ia,
                     nb::ndarray<int>& ja);

void get_state_var (nb::ndarray<double>& uv, const nb::str& name);
void set_state_var (const nb::ndarray<double>& uv, const nb::str& name);

void forward(const double dt);

void finalize();

} // namespace pyhommexx

#endif // PYHOMMEXX_UTILS_HPP
