#include "Config.hpp"
#include "Hommexx_Session.hpp"

#include <ekat_kokkos_session.hpp>

#include <iostream>

void ekat_initialize_test_session (int argc, char** argv, const bool print_config) {
  ekat::initialize_kokkos_session (argc,argv,print_config);
  Homme::Session::m_inited = true;
  Homme::Session::m_handle_kokkos = false;
  if (print_config) {
    std::cout << Homme::homme_config_settings_string ();
  }
}

void ekat_finalize_test_session () {
  ekat::finalize_kokkos_session ();
}
