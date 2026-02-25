#include "pyhommexx.hpp"

#include "Context.hpp"
#include "Hommexx_Session.hpp"
#include "Types.hpp"

#include <ekat_assert.hpp>

#include <nanobind/ndarray.h>

#include <fstream>

extern "C" {
void init_parallel_f90 ();
void print_to_screen_f90 (const bool enabled);
void prim_finalize_f90 ();
}

namespace pyhommexx {

using namespace Homme;

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

void init_session (const bool do_print_to_screen)
{
  print_to_screen(do_print_to_screen);
  init_parallel_f90();
  // Throw, so we can use try blocks in py
  Session::m_throw_instead_of_abort = true;
}

void enable_scalar_type (const nb::str& dtype)
{
  std::string dtype_str(dtype.c_str());

  if (dtype_str=="real") {
    Session::is_st_enabled<Real>() = true;
  } else if (dtype_str=="dpfad") {
#ifdef HOMMEXX_ENABLE_FAD_TYPES
    Session::is_st_enabled<DpFadType>() = true;
#else
    EKAT_ERROR_MSG("[pyhommexx] dpfad data type requires homme to be build with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
  } else {
    EKAT_ERROR_MSG("[pyhommexx] Error! Unrecognized/unsupported dtype name.\n"
        " - input dtype: " + dtype_str + "\n"
        " - valid dtype(s): real, dpfad\n");
  }
}

void print_to_screen (const bool enabled)
{
  print_to_screen_f90 (enabled);
  OutputRedirection::instance().toggle(enabled);
}

void finalize()
{
  prim_finalize_f90();
}

} // namespace pyhommexx
