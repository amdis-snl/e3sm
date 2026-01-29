#ifndef PYHOMMEXX_DECL_HPP
#define PYHOMMEXX_DECL_HPP

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace pyhommexx {

namespace nb = nanobind;

template<int N>
using ndarray_t = nb::ndarray<double,nb::ndim<N>>;

void init_session (const bool do_print_to_screen = true);

void print_to_screen (const bool enabled);

void read_params (const nb::str& nml_filename);

nb::dict get_params();
void set_params(const nb::dict& params);
int get_nelemd();

void model_init ();

void get_num_unique_pts (nb::ndarray<int>& n);
void get_unique_pts (nb::ndarray<int>& ia,
                     nb::ndarray<int>& ja);

void get_state_var (nb::ndarray<double>& uv, const nb::str& name);
void set_state_var (const nb::ndarray<double>& uv, const nb::str& name);

void run_functor(const nb::str& name,const nb::dict& params);
void forward(const double dt);

void finalize();

} // namespace pyhommexx

#endif // PYHOMMEXX_UTILS_HPP
