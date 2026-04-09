/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "Config.hpp"
#include "Hommexx_Session.hpp"
#include "ExecSpaceDefs.hpp"
#include "Types.hpp"
#include "profiling.hpp"

#include "Context.hpp"

#include "vector/vector_pragmas.hpp"

#ifdef HOMMEXX_ENABLE_FAD_TYPES
#include "Sacado_Version.hpp"
#endif

#include <ekat_comm.hpp>
#include <ekat_arch.hpp>
#include <ekat_kokkos_session.hpp>

#include <iostream>
#include <fstream>

namespace Homme
{

struct OutputRedirection {
  void toggle (bool on) {
    if (on) {
      std::cout.rdbuf(cout);
      std::cerr.rdbuf(cerr);
    } else {
      std::cout.rdbuf(blackhole.rdbuf());
      std::cerr.rdbuf(blackhole.rdbuf());
    }
  }

  static OutputRedirection& instance() {
    static OutputRedirection out_red;
    return out_red;
  }
protected:
  OutputRedirection ()
   : cout (std::cout.rdbuf())
   , cerr (std::cerr.rdbuf())
   , blackhole("/dev/null")
  {}

  std::streambuf* cout;
  std::streambuf* cerr;

  std::ofstream blackhole;
};

// Default settings for homme's session
bool Session::m_inited = false;
bool Session::m_handle_kokkos = true;
bool Session::m_screen_output_enabled = true;

void Session::toggle_screen_output (const bool enabled)
{
  OutputRedirection::instance().toggle(enabled);
  m_screen_output_enabled = enabled;
}

void print_homme_config_settings () {
  // Print configure-time settings.
#ifdef HOMMEXX_SHA1
  std::cout << "HOMMEXX SHA1: " << HOMMEXX_SHA1 << "\n";
#endif
  std::cout << "HOMMEXX VECTOR_SIZE: " << VECTOR_SIZE << "\n";
  std::cout << "HOMMEXX Active AVX settings: " + ekat::active_avx_string () + "\n";
  std::cout << "HOMMEXX MPI_ON_DEVICE: " << HOMMEXX_MPI_ON_DEVICE << "\n";
#ifdef HOMMEXX_CUDA_SHARE_BUFFER
  std::cout << "HOMMEXX CUDA_SHARE_BUFFER: on\n";
#else
  std::cout << "HOMMEXX CUDA_SHARE_BUFFER: off\n";
#endif
  std::cout << "HOMMEXX CUDA_(MIN/MAX)_WARP_PER_TEAM: " << HOMMEXX_CUDA_MIN_WARP_PER_TEAM
              << " / " << HOMMEXX_CUDA_MAX_WARP_PER_TEAM << "\n";
#ifndef HOMMEXX_NO_VECTOR_PRAGMAS
  std::cout << "HOMMEXX has vector pragmas\n";
#else
  std::cout << "HOMMEXX doesn't have vector pragmas\n";
#endif
#ifdef HOMMEXX_ENABLE_FAD_TYPES
  std::cout << "HOMMEXX has Sacado's FAD type support. " << Sacado_Version() << "\n";
  std::cout << "HOMMEXX DP_SFAD length: " << HOMMEXX_DP_SFAD_SIZE << "\n";
#else
  std::cout << "HOMME does NOT have Sacado's FAD type support\n";
#endif

#ifdef HOMMEXX_CONFIG_IS_CMAKE
  std::cout << "HOMMEXX configured with CMake\n";
# ifdef HAVE_CONFIG_H
  std::cout << "HOMMEXX has config.h.c\n";
# endif
#else
  std::cout << "HOMMEXX provided best default values in Config.hpp\n";
#endif
}

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
      print_homme_config_settings ();
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
