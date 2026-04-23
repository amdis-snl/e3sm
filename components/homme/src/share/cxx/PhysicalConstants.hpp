/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_PHYSICAL_CONSTANTS_HPP
#define HOMMEXX_PHYSICAL_CONSTANTS_HPP

#include "Types.hpp"

namespace Homme
{

namespace PhysicalConstants
{
  constexpr Real Rwater_vapor  = 461.5;
  constexpr Real Cpwater_vapor = 1870.0;
  constexpr Real Rgas          = 287.04;
  constexpr Real cp            = 1005.0;
  constexpr Real kappa         = Rgas / cp;
// real Earth
  constexpr Real rearth0       = 6.376e6;
  constexpr Real rrearth0      = 1.0 / rearth0;
  constexpr Real g             = 9.80616;
  constexpr Real p0            = 100000;         // [mbar]

  constexpr Real Tref          = 288;

  constexpr Real Tref_lapse_rate = 0.0065;
}

// This class simply provides the constants as device-friendly functions
// Other classes (like EOS) can move to require a template arg, which tells
// the compiler "where" to grab constants from. This allows us to swap in
// the constants provider, so that we can for instance test sensitivities w.r.t
// a physics constant. Of course, if a functor uses physics constants and we
// want to test its sens calculation using the phys constants as param,
// the functor impl must use a different instantiation of the constants provider.
// A possibility is to template on the provider type, and store an instance
// of the provider. Sensitivity tests can then instantiate the functor with
// an ad-hoc class that provides a way to perturb some constants

struct PhysicalConstantsProvider {
  // Thermodynamics constants
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real Rwater_vapor  () const { return PhysicalConstants::Rwater_vapor;  }
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real Cpwater_vapor () const { return PhysicalConstants::Cpwater_vapor; }
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real Rgas          () const { return PhysicalConstants::Rgas;          }
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real cp            () const { return PhysicalConstants::cp;            }
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real kappa         () const { return PhysicalConstants::kappa;         }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real Tref            () const { return PhysicalConstants::Tref;            }
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real Tref_lapse_rate () const { return PhysicalConstants::Tref_lapse_rate; }

  // Earth constants
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real rearth0  () const { return PhysicalConstants::rearth0;  }
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real rrearth0 () const { return PhysicalConstants::rrearth0; }
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real g        () const { return PhysicalConstants::g;        }
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real p0       () const { return PhysicalConstants::p0;       }  // [mbar]
};

} // namespace Homme

#endif // HOMMEXX_PHYSICAL_CONSTANTS_HPP
