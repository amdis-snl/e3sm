#include "pyhommexx.hpp"

#include "Context.hpp"
#include "CaarFunctor.hpp"
#include "ElementsState.hpp"
#include "ErrorDefs.hpp"
#include "Hommexx_Session.hpp"
#include "RKStageData.hpp"
#include "TimeLevel.hpp"
#include "Types.hpp"

#include <nanobind/ndarray.h>

#include <ekat_assert.hpp>

namespace pyhommexx {

using namespace Homme;

template<typename ST>
void run_caar_functor (const nb::dict& params)
{
  auto& c = Context::singleton();
  auto& tl = c.get<TimeLevel>();

  // Set RK stage parameters, then modify if input dict contains some
  RKStageData data;
  data.nm1 = tl.nm1;
  data.n0  = tl.n0;
  data.np1 = tl.np1;
  data.n0_qdp = 0;
  data.scale1 = 1;
  data.scale2 = 0;
  data.scale3 = 1;
  data.eta_ave_w = 0;

  for (const auto& [key, value] : params) {
    EKAT_REQUIRE_MSG(nb::isinstance<nb::str>(key),
        "Error! Functor params dict keys should be strings.\n");

    std::string key_str(nb::cast<nb::str>(key).c_str());
    if (key_str=="dt") {
      EKAT_REQUIRE_MSG(nb::isinstance<double>(value),
          "Error! Functor param 'dt' should be a double.\n");
      data.dt = nb::cast<double>(value);
    } else if (key_str=="eta_ave_w") {
      EKAT_REQUIRE_MSG(nb::isinstance<double>(value),
          "Error! Functor param 'eta_ave_w' should be a double.\n");
      data.eta_ave_w = nb::cast<double>(value);
    } else if (key_str=="scale1") {
      EKAT_REQUIRE_MSG(nb::isinstance<double>(value),
          "Error! Functor param 'scale1' should be a double.\n");
      data.scale1 = nb::cast<double>(value);
    } else if (key_str=="scale2") {
      EKAT_REQUIRE_MSG(nb::isinstance<double>(value),
          "Error! Functor param 'scale2' should be a double.\n");
      data.scale2 = nb::cast<double>(value);
    } else if (key_str=="scale3") {
      EKAT_REQUIRE_MSG(nb::isinstance<double>(value),
          "Error! Functor param 'scale3' should be a double.\n");
      data.scale3 = nb::cast<double>(value);
    } else {
      EKAT_ERROR_MSG("[ERROR] Invalid key for caar functor params.\n"
          " - key: " + key_str + "\n" +
          " - valid keys: dt, eta_ave_w, scale1, scale2, scale3\n");
    }
  }

  auto& f = c.get<CaarFunctorST<ST>>();
  auto& state = c.get<ElementsStateST<ST>>();
  f.run(data);
}

void run_functor(const nb::str& name, const nb::dict& params, const nb::str& dtype)
{
  std::string dtype_str(dtype.c_str());

  std::string name_str (name.c_str());
  if (name_str=="caar") {
    std::cout << "running functor " << name_str << ", with dtype=" << dtype_str << "\n";

    if (dtype_str=="real") {
      run_caar_functor<Real>(params);
    } else if (dtype_str=="dpfad") {
#ifdef HOMMEXX_ENABLE_FAD_TYPES
      run_caar_functor<DpFadType>(params);
#else
      EKAT_ERROR_MSG("[pyhommexx] dpfad data type requires homme to be build with HOMMEXX_ENABLE_FAD_TYPES=ON.\n");
#endif
    } else {
      EKAT_ERROR_MSG("[run_functor] Error! Unrecognized/unsupported dtupe name.\n"
          " - input dtype: " + dtype_str + "\n"
          " - valid dtype(s): real, dpfad\n");
    }
  } else {
    EKAT_ERROR_MSG("[run_functor] Unrecognized/unsupported fucntor name.\n"
        " - input name: " + name_str + "\n"
        " - valid name(s): caar\n");
  }
}

} // namespace pyhommexx
