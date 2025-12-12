#ifndef PYHOMMEXX_UTILS_HPP
#define PYHOMMEXX_UTILS_HPP

#include <nanobind/nanobind.h>
#include <mpi4py/mpi4py.h>
#include <mpi.h>

#include <typeinfo>

namespace pyhommexx
{

namespace nb = nanobind;

MPI_Comm get_c_comm (nb::object py_comm)
{
  if (import_mpi4py() < 0) {
    throw nb::python_error();
  }
  auto py_src = py_comm.ptr();
  if (not PyObject_TypeCheck(py_src, &PyMPIComm_Type)) {
    throw std::bad_cast();
  }

  auto comm_ptr = PyMPIComm_Get(py_src);
  return *comm_ptr;
}

const char* get_c_string(nb::object py_str) {
  // Ensure the Python string is of the correct type
  if (!PyUnicode_Check(py_str.ptr())) {
    throw std::bad_cast();
  }

  // Convert the Python string to a C string
  PyObject* py_bytes = PyUnicode_AsEncodedString(py_str.ptr(), "utf-8", "Error");
  if (py_bytes == nullptr) {
    throw std::runtime_error("Failed to convert Python string to bytes.");
  }

  // Get the C string from the bytes object
  const char* c_str = PyBytes_AsString(py_bytes);
  if (c_str == nullptr) {
    Py_DECREF(py_bytes);
    throw std::runtime_error("Failed to convert bytes to C string.");
  }

  // Decrease the reference count of the bytes object
  Py_DECREF(py_bytes);

  return c_str; // Return the C string
}

} // namespace pyhommexx

#endif // PYHOMMEXX_UTILS_HPP
