#include <catch2/catch.hpp>

#include "PhysicalConstants.hpp"
#include "EquationOfState.hpp"

#include "utilities/TestUtils.hpp"

#include <ekat_string_utils.hpp>

#include <algorithm>   // std::min_element
#include <cmath>       // std::abs
#include <iostream>    // std::cout
#include <limits>      // std::numeric_limits
#include <random>      // std::random_device, std::mt19937_64, std::uniform_real_distribution
#include <vector>      // std::vector

namespace Homme {

namespace PC = PhysicalConstants;

// A constants provider that perturbs Rgas by a factor (1+perturb).
// kappa = Rgas/cp is also perturbed as a result.
// p0 and cp are left unchanged.
// The perturbation is a static member so it works inside KOKKOS_INLINE_FUNCTION.
template<typename ST>
struct PerturbedConstants {

  PerturbedConstants() { perturb = ST(0); }

  static ST perturb;

  // Only decl the ones used in EOS
  static KOKKOS_FORCEINLINE_FUNCTION ST             Rgas  () { return (1+perturb)*PC::Rgas; }
  static KOKKOS_FORCEINLINE_FUNCTION constexpr Real cp    () { return PC::cp;           }
  static KOKKOS_FORCEINLINE_FUNCTION ST             kappa () { return Rgas() / cp();    }
  static KOKKOS_FORCEINLINE_FUNCTION constexpr Real p0    () { return PC::p0;           }  // [mbar]
};

template<typename ST>
ST PerturbedConstants<ST>::perturb; // Must instantiate static member vars

template<typename Provider>
using EOS = EquationOfState<Provider>;

// Helper: compute FAD derivative and value for a scalar EOS function,
// and compare against finite differences as the perturbation h is refined.
// Returns true if the minimum finite-diff error is small enough.
template<typename Func>
bool check_fad_vs_fd (Func&& run,
                      Real fad_val, Real fad_deriv,
                      const std::vector<Real>& h_vals,
                      const char* name)
{
  // baseline: run with perturb=0
  PerturbedConstants<Real>::perturb = 0.0;
  const Real f0 = run(0.0);

  // Check FAD value matches the baseline
  if (std::abs(fad_val - f0) > 1e-14 * std::abs(f0) + 1e-14) {
    std::cout << name << ": FAD val=" << fad_val << " != baseline=" << f0 << "\n";
    return false;
  }

  std::vector<Real> fd_errors;
  for (Real h : h_vals) {
    const Real fh = run(h);
    const Real fd = (fh - f0) / h;
    fd_errors.push_back(std::abs(fd - fad_deriv));
  }

  std::cout << name << ": FAD deriv = " << fad_deriv << "\n";
  std::cout << "  h   = [" << ekat::join(h_vals,",") << "]\n";
  std::cout << "  |FD-FAD| = [" << ekat::join(fd_errors,",") << "]\n";

  // Require first-order convergence: the minimum FD error must be at least
  // 100x smaller than the coarsest-h error. With h_vals spanning 1e-2 to 1e-8,
  // the truncation error drops by ~1e4 before round-off dominates, so the
  // minimum is well below fd_errors[0]/100.
  // Special case: if the coarsest-h FD error is already at machine epsilon
  // (FAD and FD agree to floating-point precision at h[0]), there is nothing
  // to converge from - treat as success.
  const Real min_err = *std::min_element(fd_errors.begin(),fd_errors.end());
  return (fd_errors[0] < std::numeric_limits<Real>::epsilon()) || (min_err < fd_errors[0] / 1e2);
}

TEST_CASE("eos_dp_check") {

  // The random numbers generator
  std::random_device rd;
  using rngAlg = std::mt19937_64;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");
  rngAlg engine(seed);
  using RPDF = std::uniform_real_distribution<Real>;

  // Create and init hvcoord (needed for EOS init, even though static scalar methods don't use it)
  HybridVCoord hvcoord;
  hvcoord.random_init(seed);

  using FadT = SFadN<Real,1>;

  // Seed the FAD perturbation: perturb has value 0, derivative 1.
  // The constructor call initialises the static perturb member to ST(0).
  (void)PerturbedConstants<FadT>{};
  PerturbedConstants<FadT>::perturb.fastAccessDx(0) = 1.0;

  // Real provider: constructor initialises static perturb to 0.
  (void)PerturbedConstants<Real>{};

  // Step sizes for finite differences: from coarse to fine to probe convergence
  const std::vector<Real> h_vals = {1e-2, 1e-4, 1e-6, 1e-8};

  SECTION ("pressure_to_exner") {
    RPDF pdf(10, 1000);
    const Real p0 = pdf(engine);

    // Run EOS with Fad type to get FAD derivative
    FadT p_sfad = p0;
    EOS<PerturbedConstants<FadT>>::pressure_to_exner(p_sfad);
    const Real fad_val   = p_sfad.val();
    const Real fad_deriv = p_sfad.fastAccessDx(0);

    auto run = [&](Real h) {
      PerturbedConstants<Real>::perturb = h;
      Real p = p0;
      EOS<PerturbedConstants<Real>>::pressure_to_exner(p);
      PerturbedConstants<Real>::perturb = 0.0;
      return p;
    };

    REQUIRE(check_fad_vs_fd(run, fad_val, fad_deriv, h_vals, "pressure_to_exner"));
  }

  SECTION ("pressure_to_recip_exner") {
    RPDF pdf(10, 1000);
    const Real p0 = pdf(engine);

    FadT p_sfad = p0;
    EOS<PerturbedConstants<FadT>>::pressure_to_recip_exner(p_sfad);
    const Real fad_val   = p_sfad.val();
    const Real fad_deriv = p_sfad.fastAccessDx(0);

    auto run = [&](Real h) {
      PerturbedConstants<Real>::perturb = h;
      Real p = p0;
      EOS<PerturbedConstants<Real>>::pressure_to_recip_exner(p);
      PerturbedConstants<Real>::perturb = 0.0;
      return p;
    };

    REQUIRE(check_fad_vs_fd(run, fad_val, fad_deriv, h_vals, "pressure_to_recip_exner"));
  }

  SECTION ("compute_dphi") {
    RPDF pdf(1, 100);
    const Real vtheta_dp = pdf(engine);
    const Real p_in      = pdf(engine);

    // FAD version: inputs are plain (no seed), only constants carry the Fad derivative
    const FadT vtheta_dp_fad = vtheta_dp;
    const FadT p_fad         = p_in;
    const FadT dphi_sfad     = EOS<PerturbedConstants<FadT>>::compute_dphi(vtheta_dp_fad, p_fad);
    const Real fad_val   = dphi_sfad.val();
    const Real fad_deriv = dphi_sfad.fastAccessDx(0);

    auto run = [&](Real h) {
      PerturbedConstants<Real>::perturb = h;
      Real dphi = EOS<PerturbedConstants<Real>>::compute_dphi(vtheta_dp, p_in);
      PerturbedConstants<Real>::perturb = 0.0;
      return dphi;
    };

    REQUIRE(check_fad_vs_fd(run, fad_val, fad_deriv, h_vals, "compute_dphi"));
  }

  SECTION ("compute_pnh_and_exner") {
    // For compute_pnh_and_exner, we need:
    //   exner_tmp = (-Rgas)*vtheta_dp / dphi  to be > 0 so that pow is valid
    // This requires vtheta_dp > 0 and dphi < 0 (delta(phi_i) is negative since phi increases upward)
    RPDF pdf_pos(1, 100);
    RPDF pdf_neg(-100, -1);
    const Real vtheta_dp = pdf_pos(engine);
    const Real dphi      = pdf_neg(engine);

    // FAD version
    const FadT vtheta_dp_fad = vtheta_dp;
    const FadT dphi_fad      = dphi;
    FadT pnh_fad, exner_fad;
    EOS<PerturbedConstants<FadT>>::compute_pnh_and_exner(vtheta_dp_fad, dphi_fad, pnh_fad, exner_fad);
    const Real fad_pnh_val    = pnh_fad.val();
    const Real fad_exner_val  = exner_fad.val();
    const Real fad_dpnh_deps  = pnh_fad.fastAccessDx(0);
    const Real fad_dexner_deps = exner_fad.fastAccessDx(0);

    // Baseline Real values
    PerturbedConstants<Real>::perturb = 0.0;
    Real pnh_0 = 0, exner_0 = 0;
    EOS<PerturbedConstants<Real>>::compute_pnh_and_exner(vtheta_dp, dphi, pnh_0, exner_0);

    REQUIRE(std::abs(fad_pnh_val - pnh_0)   < 1e-14 * std::abs(pnh_0)   + 1e-14);
    REQUIRE(std::abs(fad_exner_val - exner_0) < 1e-14 * std::abs(exner_0) + 1e-14);

    // Finite difference for pnh
    {
      auto run_pnh = [&](Real h) {
        PerturbedConstants<Real>::perturb = h;
        Real pnh_h = 0, exner_h = 0;
        EOS<PerturbedConstants<Real>>::compute_pnh_and_exner(vtheta_dp, dphi, pnh_h, exner_h);
        PerturbedConstants<Real>::perturb = 0.0;
        return pnh_h;
      };
      REQUIRE(check_fad_vs_fd(run_pnh, fad_pnh_val, fad_dpnh_deps, h_vals, "compute_pnh_and_exner::pnh"));
    }

    // Finite difference for exner
    {
      auto run_exner = [&](Real h) {
        PerturbedConstants<Real>::perturb = h;
        Real pnh_h = 0, exner_h = 0;
        EOS<PerturbedConstants<Real>>::compute_pnh_and_exner(vtheta_dp, dphi, pnh_h, exner_h);
        PerturbedConstants<Real>::perturb = 0.0;
        return exner_h;
      };
      REQUIRE(check_fad_vs_fd(run_exner, fad_exner_val, fad_dexner_deps, h_vals, "compute_pnh_and_exner::exner"));
    }
  }

} // TEST_CASE

} // namespace Homme
