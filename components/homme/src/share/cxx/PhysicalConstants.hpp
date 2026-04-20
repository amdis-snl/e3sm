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
// the functor impl must:
//  - a) be templated on the constants provider
//  - b) change PhysicalConstants::xyz to TemplateArg::xyz()

struct PhysicalConstantsProvider {
  // Thermodynamics constants
  static KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real Rwater_vapor  () { return PhysicalConstants::Rwater_vapor;  }
  static KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real Cpwater_vapor () { return PhysicalConstants::Cpwater_vapor; }
  static KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real Rgas          () { return PhysicalConstants::Rgas;          }
  static KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real cp            () { return PhysicalConstants::cp;            }
  static KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real kappa         () { return PhysicalConstants::kappa;         }

  static KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real Tref            () { return PhysicalConstants::Tref;            }
  static KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real Tref_lapse_rate () { return PhysicalConstants::Tref_lapse_rate; }

  // Earth constants
  static KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real rearth0  () { return PhysicalConstants::rearth0;  }
  static KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real rrearth0 () { return PhysicalConstants::rrearth0; }
  static KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real g        () { return PhysicalConstants::g;        }
  static KOKKOS_FORCEINLINE_FUNCTION
  constexpr Real p0       () { return PhysicalConstants::p0;       }  // [mbar]
};

} // namespace Homme

#endif // HOMMEXX_PHYSICAL_CONSTANTS_HPP
