#include "pyhommexx.hpp"

#include "Context.hpp"
#include "Hommexx_Session.hpp"
#include "PhysicalConstants.hpp"
#include "Types.hpp"

#include <ekat_assert.hpp>

#include <nanobind/ndarray.h>

#include <limits>
#include <fstream>

extern "C" {
void init_parallel_f90 ();
void prim_finalize_f90 ();
void toggle_screen_output_f90 (const bool);
}

namespace pyhommexx {

using namespace Homme;

void init_session (const bool do_print_to_screen)
{
  toggle_screen_output(do_print_to_screen);
  init_parallel_f90();
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
    EKAT_ERROR_MSG("[pyhommexx] dpfad data type requires homme to be built with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
  } else {
    EKAT_ERROR_MSG("[pyhommexx] Error! Unrecognized/unsupported dtype name.\n"
        " - input dtype: " + dtype_str + "\n"
        " - valid dtype(s): real, dpfad\n");
  }
}

void toggle_screen_output (const bool enabled)
{
  Session::toggle_screen_output (enabled);
  toggle_screen_output_f90 (enabled);
}

double get_phys_constant (const nb::str& name)
{
  std::string name_str(name.c_str());

  if (name_str=="rearth") {
    return PhysicalConstants::rearth0;
  } else if (name_str=="rgas") {
    return PhysicalConstants::Rgas;
  } else {
      EKAT_ERROR_MSG(
          "[get_phys_constant] Error! Unrecognized/unsupported constant name.\n"
          " - input name: " + name_str + "\n"
          " - valid name(s): rearth, rgas\n");
  }

  return std::numeric_limits<Real>::quiet_NaN();
}

void finalize()
{
  prim_finalize_f90();
}

} // namespace pyhommexx
