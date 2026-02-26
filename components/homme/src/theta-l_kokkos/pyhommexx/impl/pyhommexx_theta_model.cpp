#include "pyhommexx.hpp"

#include "Types.hpp"

#include <nanobind/ndarray.h>

extern "C" {
namespace Homme {
void prim_run_subcycle_c (const Real& dt, int& nstep, int& nm1, int& n0, int& np1,
                          const int& next_output_step, const int& nsplit_iteration);
}
void model_init_f90 ();
}

namespace pyhommexx {

using namespace Homme;

void model_init ()
{
  model_init_f90();
}

void forward(const double dt)
{
  int nm1, n0, np1, nstep;
  prim_run_subcycle_c(dt,nstep,nm1,n0,np1,-1,1);
}

} // namespace pyhommexx
