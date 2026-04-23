#include <catch2/catch.hpp>

#include "Types.hpp"
#include "utilities/scream_tridiag.hpp"

#include <ekat_string_utils.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace Homme {

TEST_CASE ("tridiag_dp_testing")
{
  // Setup random number generator
  std::random_device rd;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");

  using rngAlg = std::mt19937_64;
  using rpdf = std::uniform_real_distribution<Real>;
  rngAlg engine(seed);

  // Keep dimensions modest for fast unit tests, but use multiple rhs columns.
  constexpr int nrow = 16;
  constexpr int nrhs = 3;
  const std::vector<Real> h_vals = {1e-2, 1e-4, 1e-6, 1e-8};
  const Real rtol = 1e-2;
  const Real atol = 1e-12;

  using View1D = Kokkos::View<Real*, HostMemSpace>;
  using View2D = Kokkos::View<Real**, HostMemSpace>;
  using FadView1D = Kokkos::View<DpFadType*, HostMemSpace>;
  using FadView2D = Kokkos::View<DpFadType**, HostMemSpace>;

  View1D dl0("dl0", nrow), d0("d0", nrow), du0("du0", nrow);
  View2D rhs0("rhs0", nrow, nrhs), drhs("drhs", nrow, nrhs);

  for (int i = 0; i < nrow; ++i) {
    dl0(i) = i == 0 ? 0.0 : rpdf(0.05,0.2)(engine);
    du0(i) = i == nrow-1 ? 0.0 : rpdf(0.05,0.2)(engine);
    d0(i) = 2.0 + std::abs(dl0(i)) + std::abs(du0(i));
    for (int j = 0; j < nrhs; ++j) {
      rhs0(i,j) = rpdf(-1,1)(engine);
      drhs(i,j) = rpdf(-1,1)(engine);
    }
  }

  auto copy_diag = [&](View1D dl, View1D d, View1D du) {
    for (int i = 0; i < nrow; ++i) {
      dl(i) = dl0(i);
      d(i) = d0(i);
      du(i) = du0(i);
    }
  };

  auto solve_real = [&](Real h, View2D x) {
    View1D dl("dl", nrow), d("d", nrow), du("du", nrow);
    copy_diag(dl,d,du);
    for (int i = 0; i < nrow; ++i) {
      for (int j = 0; j < nrhs; ++j) {
        x(i,j) = rhs0(i,j) + h*drhs(i,j);
      }
    }
    scream::tridiag::thomas(dl,d,du,x);
  };

  View2D x0("x0", nrow, nrhs);
  solve_real(0.0, x0);

  FadView1D dl_fad("dl_fad", nrow), d_fad("d_fad", nrow), du_fad("du_fad", nrow);
  FadView2D x_fad("x_fad", nrow, nrhs);
  for (int i = 0; i < nrow; ++i) {
    dl_fad(i) = dl0(i);
    d_fad(i) = d0(i);
    du_fad(i) = du0(i);
  }
  DpFadType p_fad = DpFadType(0);
  p_fad.fastAccessDx(0) = 1.0;
  for (int i = 0; i < nrow; ++i) {
    for (int j = 0; j < nrhs; ++j) {
      x_fad(i,j) = rhs0(i,j) + p_fad*drhs(i,j);
    }
  }
  scream::tridiag::thomas(dl_fad,d_fad,du_fad,x_fad);

  std::vector<Real> eh;
  for (auto h : h_vals) {
    View2D xh("xh", nrow, nrhs);
    solve_real(h, xh);
    Real linf = 0;
    for (int i = 0; i < nrow; ++i) {
      for (int j = 0; j < nrhs; ++j) {
        const Real dfdp_fd = (xh(i,j) - x0(i,j))/h;
        const Real dfdp_ad = x_fad(i,j).fastAccessDx(0);
        linf = std::max(linf, std::abs(dfdp_fd - dfdp_ad));
      }
    }
    eh.emplace_back(linf);
  }

  std::cout << std::setprecision(16) << "      h = [" << ekat::join(h_vals,",") << "]\n";
  std::cout << std::setprecision(16) << "      |dx/dp - FD(dxdp)|_inf = [" << ekat::join(eh,",") << "]\n";
  // Accept if: (a) coarsest-step error is already absolutely tiny, or
  // (b) refinement decreases the error substantially before roundoff dominates.
  const Real min_err = *std::min_element(eh.begin(),eh.end());
  REQUIRE ((eh[0]<atol or min_err<eh[0]*rtol));
}

} // namespace Homme
