#include "pyhommexx_utils.hpp"

#include <mpi4py/mpi4py.h>
#include <mpi.h>

#include <typeinfo>

namespace pyhommexx {

int get_f_comm (nb::object py_comm)
{
  if (import_mpi4py() < 0) {
    throw nb::python_error();
  }
  auto py_src = py_comm.ptr();
  if (not PyObject_TypeCheck(py_src, &PyMPIComm_Type)) {
    throw std::bad_cast();
  }

  auto comm_ptr = PyMPIComm_Get(py_src);
  return MPI_Comm_c2f(*comm_ptr);
}

} // namespace pyhommexx
