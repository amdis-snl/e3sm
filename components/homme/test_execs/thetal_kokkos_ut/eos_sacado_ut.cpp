#include <catch2/catch.hpp>

#include "PhysicalConstants.hpp"
#include "EquationOfState.hpp"

#include "utilities/TestUtils.hpp"

#include <ekat_string_utils.hpp>

namespace Homme {

namespace PC = PhysicalConstants;

template<typename ST>
struct PerturbedConstants {

  PerturbedConstants() { perturb = ST(0); }

  static ST perturb;

  // Only decl the ones used in EOS
  static KOKKOS_FORCEINLINE_FUNCTION constexpr ST   Rgas  () { return (1+perturb)*PC::Rgas; }
  static KOKKOS_FORCEINLINE_FUNCTION constexpr Real cp    () { return PC::cp;           }
  static KOKKOS_FORCEINLINE_FUNCTION constexpr ST   kappa () { return Rgas() / cp();    }
  static KOKKOS_FORCEINLINE_FUNCTION constexpr Real p0    () { return PC::p0;           }  // [mbar]
};

template<typename ST>
ST PerturbedConstants<ST>::perturb; // Must instantiate static member vars

template<typename Provider>
using EOS = EquationOfState<Provider>;

TEST_CASE("eos_dp_check") {

  // The random numbers generator
  std::random_device rd;
  using rngAlg = std::mt19937_64;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");
  rngAlg engine(seed);
  using RPDF = std::uniform_real_distribution<Real>;

  // Create and init hvcoord
  HybridVCoord hvcoord;
  hvcoord.random_init(seed);

  using FadT = SFadN<Real,1>;

  PerturbedConstants<Real> provider_real;
  PerturbedConstants<FadT> provider_sfad;
  provider_sfad.perturb.fastAccessDx(0) = 1;
  for (bool hydrostatic : {true, false}) {
    // Create the EOS objects
    EOS<PerturbedConstants<Real>> eos_real;
    EOS<PerturbedConstants<FadT>> eos_sfad;

    eos_real.init(hydrostatic,hvcoord);
    eos_sfad.init(hydrostatic,hvcoord);

    SECTION ("pressure_to_exner") {

      RPDF pdf(10,1000);

      Real p_real0 = pdf(engine);
      FadT p_sfad  = p_real0;

      // Run eos_sfad::pressure_to_exner
      // Run eos_real::pressure_to_exner for provider.perturb=0
      // Run eos_real::pressure_to_exner for provider.perturb=1,0.1,0.01,0.001,...
      // Compute finite-diff deriv dexner/dperturb, compare with Fad deriv
    }
    // Add similar sections for other EquationOfState methods
  }

}

} // namespace Homme
