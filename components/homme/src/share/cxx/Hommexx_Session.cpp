/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "Config.hpp"
#include "Hommexx_Session.hpp"
#include "HommeExecSpace.hpp"
#include "profiling.hpp"

#include "Context.hpp"

#ifdef HOMMEXX_ENABLE_FAD_TYPES
#include "Sacado_Version.hpp"
#endif

#include <ekat_comm.hpp>
#include <ekat_kokkos_session.hpp>

#include <iostream>
#include <fstream>

namespace Homme
{

// Instantiate Session's static members
bool Session::m_inited = false;
bool Session::m_handle_kokkos = true;
bool Session::m_screen_output_enabled = true;

void initialize_hommexx_session ()
{
  // If hommexx session is not currently inited, then init it.
  if (!Session::m_inited) {
    /* Make certain profiling is only done for code we're working on */
    profiling_pause();

    /* Set Environment variables to control how many
     * threads/processors Kokkos uses */
    if (Session::m_handle_kokkos) {
      ekat::initialize_kokkos_session(false);
    }

    // Note: at this point, the Comm *should* already be created.
    const auto& comm = Context::singleton().get<ekat::Comm>();
    if (comm.am_i_root()) {
      ExecSpace().print_configuration(std::cout, true);
      std::cout << homme_config_settings_string () << "\n";
    }

    Session::m_inited = true;

    // Make sure the default scalar type is marked as "enabled" in the context
    Session::is_st_enabled<ScalarValue>() = true;
  }
}

int get_hommexx_dp_sfad_size ()
{
#ifdef HOMMEXX_ENABLE_FAD_TYPES
  return HOMMEXX_DP_SFAD_SIZE;
#else
  return 0;
#endif
}

void finalize_hommexx_session ()
{
  if (Session::m_inited) {
    Context::finalize_singleton();

    if (Session::m_handle_kokkos) {
      ekat::finalize_kokkos_session();
    }
  }

  Session::m_inited = false;
}

} // namespace Homme
