/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_EXEC_SPACE_HPP
#define HOMMEXX_EXEC_SPACE_HPP

#include <Kokkos_Core.hpp>

#include "Config.hpp"

namespace Homme
{

// Some in-house names for Kokkos exec spaces, which are
// always defined, possibly as alias of void

#ifdef HOMMEXX_ENABLE_GPU

#ifdef KOKKOS_ENABLE_CUDA
using HommexxGPU = Kokkos::Cuda;
#endif

#ifdef KOKKOS_ENABLE_HIP
using HommexxGPU = Kokkos::Experimental::HIP;
#endif

#ifdef KOKKOS_ENABLE_SYCL
using HommexxGPU = Kokkos::Experimental::SYCL;
#endif

#else
using HommexxGPU = void;
#endif

#ifdef KOKKOS_ENABLE_OPENMP
using Hommexx_OpenMP = Kokkos::OpenMP;
#else
using Hommexx_OpenMP = void;
#endif

#ifdef KOKKOS_ENABLE_PTHREADS
using Hommexx_Threads = Kokkos::Threads;
#else
using Hommexx_Threads = void;
#endif

#ifdef KOKKOS_ENABLE_SERIAL
using Hommexx_Serial = Kokkos::Serial;
#else
using Hommexx_Serial = void;
#endif

#ifdef HOMMEXX_ENABLE_GPU
# define HOMMEXX_STATIC
#else
# define HOMMEXX_STATIC static
#endif

// Selecting the execution space. If no specific request, use Kokkos default
// exec space
#ifdef HOMMEXX_ENABLE_GPU
using ExecSpace = HommexxGPU;
#elif defined(HOMMEXX_OPENMP_SPACE)
using ExecSpace = Hommexx_OpenMP;
#elif defined(HOMMEXX_THREADS_SPACE)
using ExecSpace = Hommexx_Threads;
#elif defined(HOMMEXX_SERIAL_SPACE)
using ExecSpace = Hommexx_Serial;
#elif defined(HOMMEXX_DEFAULT_SPACE)
using ExecSpace = Kokkos::DefaultExecutionSpace::execution_space;
#else
#error "No valid execution space choice"
#endif // HOMMEXX_EXEC_SPACE

static_assert (!std::is_same<ExecSpace,void>::value,
               "Error! You are trying to use an ExecutionSpace not enabled in Kokkos.\n");

// A templated typedef for MD range policy
template<typename ExecutionSpace, int Rank>
using MDRangePolicy = Kokkos::MDRangePolicy
                          < ExecutionSpace,
                            Kokkos::Rank
                              < Rank,
                                Kokkos::Iterate::Right,
                                Kokkos::Iterate::Right
                              >,
                            Kokkos::IndexType<int>
                          >;

template <typename ExeSpace>
struct OnGpu { enum : bool { value = false }; };

template <>
struct OnGpu<HommexxGPU> { enum : bool { value = true }; };

} // namespace Homme

#endif // HOMMEXX_EXEC_SPACE_HPP
