/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_CONFIG_HPP
#define HOMMEXX_CONFIG_HPP

#ifdef HOMMEXX_CONFIG_IS_CMAKE
# include "Hommexx_config.h"
# ifdef HAVE_CONFIG_H
#  include "config.h.c"
# endif
#elif !defined(HOMMEXX_VECTOR_SIZE) 
// Establish a good candidate vector size for eam builds
# ifdef HOMMEXX_ENABLE_GPU
#  define HOMMEXX_VECTOR_SIZE 1
# else
#  define HOMMEXX_VECTOR_SIZE 8
# endif
#endif

#if ! defined HOMMEXX_CUDA_SPACE && ! defined HOMMEXX_OPENMP_SPACE && ! defined HOMMEXX_THREADS_SPACE && ! defined HOMMEXX_SERIAL_SPACE && ! defined HOMMEXX_HIP_SPACE && ! defined HOMMEXX_SYCL_SPACE
# define HOMMEXX_DEFAULT_SPACE
#endif

#ifndef HOMMEXX_MPI_ON_DEVICE
# define HOMMEXX_MPI_ON_DEVICE 1
#endif

#include <Kokkos_Core.hpp>

#ifdef HOMMEXX_ENABLE_GPU 
# ifndef HOMMEXX_CUDA_MIN_WARP_PER_TEAM
#  define HOMMEXX_CUDA_MIN_WARP_PER_TEAM 8
# endif
# ifndef HOMMEXX_CUDA_MAX_WARP_PER_TEAM
#  define HOMMEXX_CUDA_MAX_WARP_PER_TEAM 16
# endif
#elif !defined(HOMMEXX_CONFIG_IS_CMAKE)
# define HOMMEXX_CUDA_MIN_WARP_PER_TEAM 1
# define HOMMEXX_CUDA_MAX_WARP_PER_TEAM 1
#endif

#if defined KOKKOS_COMPILER_GNU
// See https://github.com/kokkos/kokkos-kernels/issues/129
# define ConstExceptGnu
#else
# define ConstExceptGnu const
#endif

#include <ekat_arch.hpp>

#ifdef HOMMEXX_ENABLE_FAD_TYPES
// Disable view specializations
#define SACADO_DISABLE_FAD_VIEW_SPEC
#include <Sacado.hpp>
#endif

#include <string>

namespace Homme {

inline std::string homme_config_settings_string ()
{
  std::string s;
  // Print configure-time settings.
#ifdef HOMMEXX_SHA1
  s += "HOMMEXX SHA1: " + std::string(HOMMEXX_SHA1) + "\n";
#endif
  s += "HOMMEXX VECTOR_SIZE: " + std::to_string(HOMMEXX_VECTOR_SIZE) + "\n";
  s += "HOMMEXX Active AVX settings: " + ekat::active_avx_string () + "\n";
  s += "HOMMEXX MPI_ON_DEVICE: " + std::string(HOMMEXX_MPI_ON_DEVICE ? "yes" : "no") + "\n";
#ifdef HOMMEXX_CUDA_SHARE_BUFFER
  s += "HOMMEXX CUDA_SHARE_BUFFER: on\n";
#else
  s += "HOMMEXX CUDA_SHARE_BUFFER: off\n";
#endif
  s += "HOMMEXX CUDA_(MIN/MAX)_WARP_PER_TEAM: "
     + std::to_string(HOMMEXX_CUDA_MIN_WARP_PER_TEAM) + "/"
     + std::to_string(HOMMEXX_CUDA_MAX_WARP_PER_TEAM) + "\n";
#ifndef HOMMEXX_NO_VECTOR_PRAGMAS
  s += "HOMMEXX has vector pragmas\n";
#else
  s += "HOMMEXX doesn't have vector pragmas\n";
#endif
#ifdef HOMMEXX_ENABLE_FAD_TYPES
  s += "HOMMEXX has Sacado's FAD type support. " + Sacado_Version() + "\n";
  s += "HOMMEXX DP_SFAD length: " + std::to_string(HOMMEXX_DP_SFAD_SIZE) + "\n";
#else
  s += "HOMME does NOT have Sacado's FAD type support\n";
#endif

#ifdef HOMMEXX_CONFIG_IS_CMAKE
  s += "HOMMEXX configured with CMake\n";
# ifdef HAVE_CONFIG_H
  s += "HOMMEXX has config.h.c\n";
# endif
#else
  s += "HOMMEXX provided best default values in Config.hpp\n";
#endif
  return s;
}

} // namespace Homme

#endif // HOMMEXX_CONFIG_HPP
