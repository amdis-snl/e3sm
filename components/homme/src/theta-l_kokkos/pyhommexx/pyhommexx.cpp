#include "pyhommexx_decl.hpp"

#include <nanobind/nanobind.h>

namespace Homme {
namespace pyhommexx {

NB_MODULE (pyhommexx,m) {

  m.doc() = "Python interface to theta-l_kokkos Hommexx target";
  m.def("init",&init);
  m.def("get_params",&get_params);
  m.def("get_num_unique_pts",&get_num_unique_pts);
  m.def("get_unique_pts",&get_unique_pts);
  m.def("get_state_var",&get_state_var);
  m.def("set_state_var",&set_state_var);
  m.def("forward",&forward);
  m.def("finalize",&finalize);
}

} // namespace pyhommexx
} // namespace 
