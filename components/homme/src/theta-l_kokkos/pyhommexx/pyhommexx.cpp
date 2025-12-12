#include "pyhommexx_utils.hpp"
#include "pyhommexx_c2f.hpp"

#include <nanobind/nanobind.h>
#include <mpi4py/mpi4py.h>
#include <mpi.h>
#include <cstdio>
#include <fstream>

namespace nb = nanobind;

namespace pyhommexx
{

void init (nb::object py_comm)
{
  const auto& comm = get_c_comm(py_comm);
  init_parallel_f90(MPI_Comm_c2f(comm));

  prim_init_f90();
}

void forward()
{
  prim_forward_f90();
}

void finalize()
{
  prim_finalize_f90();
}

NB_MODULE (pyhommexx,m) {

  m.doc() = "Python interface to theta-l_kokkos Hommexx target";
  m.def("init",&init);
  m.def("forward",&forward);
  m.def("finalize",&finalize);
}

} // namespace pyhommexx
