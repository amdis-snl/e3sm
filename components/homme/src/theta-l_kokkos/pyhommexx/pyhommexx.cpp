#include "pyhommexx_decl.hpp"

#include <nanobind/nanobind.h>

namespace pyhommexx {

NB_MODULE (pyhommexx,m) {

  m.doc() = "Python interface to theta-l_kokkos Hommexx target";
  m.def("init_session",&init_session,
        "Initialize the session. If do_print_to_screen is True, print to screen.",
        nb::arg("do_print_to_screen") = true);
  m.def("enable_scalar_type",&enable_scalar_type);
  m.def("print_to_screen",&print_to_screen);
  m.def("read_params",&read_params);
  m.def("model_init",&model_init);
  m.def("get_nelemd",&get_nelemd);
  m.def("get_params",&get_params);
  m.def("set_params",&set_params);
  m.def("get_num_unique_pts",&get_num_unique_pts);
  m.def("get_unique_pts",&get_unique_pts);
  m.def("get_state_var",&get_state_var);
  m.def("get_state_var_dp_sens",&get_state_var_dp_sens);
  m.def("copy_state",&copy_state);
  m.def("init_dp3d_from_ps",&init_dp3d_from_ps);
  m.def("set_state_var",&set_state_var);
  m.def("perturb_state_var",&perturb_state_var,
      "Perturbs a state variable by multiply by a spatially gaussian factor, centered at given lat/lon.\n"
      "The dtype arg specifies which data structure to perturb",
        nb::arg("name"),
        nb::arg("lat0"),nb::arg("lon0"),
        nb::arg("p_max"),nb::arg("sigma"),
        nb::arg("dtype") = "real");
  m.def("run_functor",&run_functor,
      "Perturbs a state variable by multiply by a spatially gaussian factor, centered at given lat/lon"
      "The dtype arg specifies which instantiation of the functor to use",
        nb::arg("name"), nb::arg("params"), nb::arg("dtype") = "real");
  m.def("forward",&forward);
  m.def("finalize",&finalize);
}

} // namespace pyhommexx
