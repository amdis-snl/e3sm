#include "pyhommexx.hpp"

#include "Context.hpp"
#include "SimulationParams.hpp"
#include "Types.hpp"

#include <nanobind/ndarray.h>

#include <ekat_assert.hpp>

extern "C"
{
void read_params_f90 (const char*& filename);
} // extern "C"

namespace pyhommexx {

void read_params (const nb::str& nml_filename)
{
  auto nml_filename_c = nml_filename.c_str();
  read_params_f90(nml_filename_c);
}

nb::dict get_params()
{
  using namespace Homme;

  const auto& c = Context::singleton();
  const auto& s = c.get<SimulationParams>();

  nb::dict params;

  params["rsplit"] = s.rsplit;
  params["qsplit"] = s.qsplit;
  params["qsize"] = s.qsize;
  params["qsize_d"] = QSIZE_D;
  params["np"] = NP;
  params["nlev"] = NUM_PHYSICAL_LEV;
  params["hydrostatic"] = s.theta_hydrostatic_mode;
  params["nu"] = s.nu;
  params["ne"] = s.ne;

  return params;
}

void set_params(const nb::dict& params)
{
  using namespace Homme;

  auto& c = Context::singleton();
  auto& s = c.get<SimulationParams>();

  for (const auto& [key,value] : params) {
    EKAT_REQUIRE_MSG (nb::isinstance<nb::str>(key),
        "[pyhommexx::set_params] Functor params dict keys should be strings.\n");

    std::string key_str(nb::cast<nb::str>(key).c_str());
    if (key_str=="alloc_sphere_coords") {
      EKAT_REQUIRE_MSG(nb::isinstance<bool>(value),
          "[pyhommexx::set_params] Functor param 'alloc_sphere_coords' should be a boolean.\n");
      s.alloc_sphere_coords = nb::cast<bool>(value);
    } else {
      EKAT_ERROR_MSG("[pyhommexx::set_params] Invalid key for set_params.\n"
          " - key: " + key_str + "\n" +
          " - valid keys: alloc_sphere_coords\n");
    }
  }
}

} // namespace pyhommexx
