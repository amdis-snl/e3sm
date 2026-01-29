#include "pyhommexx_decl.hpp"

#include <nanobind/nanobind.h>

namespace pyhommexx {

NB_MODULE (pyhommexx,m) {

  m.doc() = "Python interface to theta-l_kokkos Hommexx target";
  m.def("init_session",&init_session);
  m.def("read_params",&read_params);
  m.def("model_init",&model_init);
  m.def("get_nelemd",&get_nelemd);
  m.def("get_params",&get_params);
  m.def("get_num_unique_pts",&get_num_unique_pts);
  m.def("get_unique_pts",&get_unique_pts);
  m.def("get_state_var",&get_state_var);
  m.def("set_state_var",&set_state_var);
  m.def("run_functor",&run_functor);
  m.def("forward",&forward);
  m.def("finalize",&finalize);
}

} // namespace pyhommexx
