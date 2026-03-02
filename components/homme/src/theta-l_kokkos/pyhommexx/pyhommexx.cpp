#include "pyhommexx.hpp"

#include <nanobind/nanobind.h>

namespace pyhommexx {

NB_MODULE (pyhommexx,m) {

  m.doc() = "Python interface to theta-l_kokkos Hommexx target";

  // General session functions
  m.def("init_session",&init_session,
        "Initialize the session. If do_print_to_screen is True, print to screen.",
        nb::arg("do_print_to_screen") = true);
  m.def("enable_scalar_type",&enable_scalar_type);
  m.def("toggle_screen_output",&toggle_screen_output);
  m.def("get_phys_constant",&get_phys_constant);
  m.def("finalize",&finalize);

  // Input parameters utilities
  m.def("read_params",&read_params);
  m.def("get_params",&get_params);
  m.def("set_params",&set_params);

  // Model geometry utilities
  m.def("get_nelemd",&get_nelemd);
  m.def("get_num_unique_pts",&get_num_unique_pts);
  m.def("get_unique_pts",&get_unique_pts);
  m.def("get_dyn_latlon",&get_dyn_latlon);

  // State handling utils
  m.def("get_state_var",&get_state_var,
      "Retrieves a state variable from the data structure with the given scalar type, at the given time slice.\n"
      "The time slice value can be -1 (nm1) 0 (n0) or 1 (np1) for state vars, and 0 for tracers (no time slices).\n"
      "By default, we retrieve slice np1 for states",
        nb::arg("arr"),
        nb::arg("name"),
        nb::arg("dtype") = "real",
        nb::arg("tl") = 1);
  m.def("get_state_var_dp_sens",&get_state_var_dp_sens);
  m.def("copy_state",&copy_state);
  m.def("init_dp3d_from_ps",&init_dp3d_from_ps);
  m.def("set_state_var",&set_state_var,
      "Sets a state variable in the data structure with the given scalar type, at the given time slice.\n"
      "The time slice value can be -1 (nm1) 0 (n0) or 1 (np1) for state vars, and 0 for tracers (no time slices).\n"
      "By default, we set slice n0 for states",
        nb::arg("arr"),
        nb::arg("name"),
        nb::arg("dtype") = "real",
        nb::arg("tl") = 0);
  m.def("set_state_var_value",&set_state_var_value,
      "Sets a state variable in the data structure with the given scalar type, at the given time slice.\n"
      "The time slice value can be -1 (nm1) 0 (n0) or 1 (np1) for state vars, and 0 for tracers (no time slices).\n"
      "By default, we set slice n0 for states",
        nb::arg("value"),
        nb::arg("name"),
        nb::arg("dtype") = "real",
        nb::arg("tl") = 0);
  m.def("perturb_state_var",&perturb_state_var,
      "Perturbs a state variable by multiply by a spatially gaussian factor, centered at given lat/lon.\n"
      "The dtype arg specifies which data structure to perturb",
        nb::arg("name"),
        nb::arg("lat0"),nb::arg("lon0"),
        nb::arg("p_max"),nb::arg("sigma"),
        nb::arg("dtype") = "real");

  // Init/run a functor or the whole model
  m.def("model_init",&model_init);
  m.def("run_functor",&run_functor,
      "Runs a named functor with the given parameters and scalar type.\n"
      "The dtype arg specifies which instantiation of the functor to use",
        nb::arg("name"), nb::arg("params"), nb::arg("dtype") = "real");
  m.def("forward",&forward);
}

} // namespace pyhommexx
