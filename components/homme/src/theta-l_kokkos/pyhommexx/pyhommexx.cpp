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
  // m.def("set_state",&set_state);
  m.def("get_state",&get_state);
  m.def("forward",&forward);
  m.def("finalize",&finalize);
}

} // namespace pyhommexx
} // namespace 
