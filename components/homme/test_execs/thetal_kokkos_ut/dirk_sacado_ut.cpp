#include <catch2/catch.hpp>

#include "DirkFunctorImpl.hpp"

#include <random>

#include "Types.hpp"
#include "Context.hpp"
#include "mpi/Connectivity.hpp"
#include "SimulationParams.hpp"
#include "Elements.hpp"
#include "PhysicalConstants.hpp"

#include "utilities/TestUtils.hpp"
#include "utilities/SyncUtils.hpp"
#include "utilities/ViewUtils.hpp"

#include <ekat_string_utils.hpp>
#include <ekat_test_utils.hpp>

#include <iomanip>

using namespace Homme;

// A constants provider that perturbs Rgas by a factor (1+perturb).
// kappa = Rgas/cp is also perturbed as a result.
template<typename ST>
struct PerturbedConstants : public PhysicalConstantsProvider {
  using PC = PhysicalConstantsProvider;

  PerturbedConstants() = default;

  ST perturb = 0;

  // Only redefine the ones we want to perturb (the rest is inherited)
  KOKKOS_FORCEINLINE_FUNCTION ST Rgas  () const { return (1+perturb)*PC::Rgas(); }
  KOKKOS_FORCEINLINE_FUNCTION ST kappa () const { return Rgas() / cp();    }
};

TEST_CASE ("dirk_dp_testing")
{
  // Setup random number generator
  std::random_device rd;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");

  using rpdf = std::uniform_real_distribution<Real>;
  using ipdf = std::uniform_int_distribution<int>;
  using rngAlg = std::mt19937_64;

  rngAlg engine(seed);

  // Create data structures
  Context::finalize_singleton();
  Context::singleton().create<ekat::Comm>(MPI_COMM_WORLD);

  const auto ALL = Kokkos::ALL;
  const int ntl = NUM_TIME_LEVELS;
  const int nlev = NUM_PHYSICAL_LEV;
  const int np = NP;

  auto& ts = ekat::TestSession::get();

  int num_elems = ipdf(5,50)(engine);
  if (ts.params.count("ne")>0) {
    num_elems = std::stoi(ts.params["ne"]);
  }
  int nm1 = 0;
  int n0  = 1;
  int np1 = 2;

  HybridVCoord hvcoord;
  hvcoord.random_init(seed);
  const auto max_pressure = 1000 + hvcoord.ps0;

  ElementsST<Real> elems_h0, elems_h, elems_ref;
  ElementsST<DpFadType> elems_dp;
  elems_h0.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  elems_h.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  elems_dp.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  elems_ref.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);

  auto& geometry = elems_h0.m_geometry = elems_h.m_geometry = elems_dp.m_geometry = elems_ref.m_geometry;
  Context::singleton().create_ref(elems_h0.m_geometry);
  geometry.randomize(seed);

  elems_ref.m_state.randomize(seed,max_pressure,hvcoord.ps0,hvcoord.hybrid_ai0,geometry.m_phis);

  // Create Perturbed constants and Dirk instances
  using PerturbedConstantsReal = PerturbedConstants<Real>;
  using PerturbedConstantsFad  = PerturbedConstants<DpFadType>;
  using DirkReal = DirkFunctorImplST<Real,PerturbedConstantsReal>;
  using DirkFad  = DirkFunctorImplST<DpFadType,PerturbedConstantsFad>;

  // EquationOfState<PerturbedConstantsReal> eos_real;
  // EquationOfState<PerturbedConstantsFad>  eos_fad;

  DirkReal dirk_real(num_elems);
  DirkFad  dirk_dp(num_elems);
  dirk_real.m_verbose = false;
  dirk_dp.m_verbose = false;
  dirk_dp.m_eos.m_constants.perturb.fastAccessDx(0) = 1;

  FunctorsBuffersManager fbm;
  fbm.request_size(dirk_real.requested_buffer_size());
  fbm.request_size(dirk_dp.requested_buffer_size());
  fbm.allocate();
  dirk_real.init_buffers(fbm);
  dirk_dp.init_buffers(fbm);

  // Run for different configs and check convergence as perturb->0
  const std::vector<Real> h_vals = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8};
  Real dt2 = rpdf(0.1,0.9)(engine);
  const auto rtol = 1e-4;
  const auto atol = 1e-6;

  SECTION ("initial_guess") {
    // Run dirk_dp
    elems_dp.m_state.import_values(elems_ref.m_state,np1);
    dirk_dp.run_initial_guess(np1,elems_dp,hvcoord);
    auto dphi = ekat::scalarize(elems_dp.m_derived.m_divdp_proj);

    // Run dirk_real with no perturbation
    dirk_real.m_eos.m_constants.perturb = 0;
    elems_h0.m_state.import_values(elems_ref.m_state,np1);
    dirk_real.run_initial_guess(np1,elems_h0,hvcoord);
    auto phi0 = ekat::scalarize(elems_h0.m_derived.m_divdp_proj);

    Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>> policy({0,0,0,0},{num_elems,np,np,nlev});
    // Run perturbed dirk_real
    std::vector<Real> eh;
    for (auto h : h_vals) {
      dirk_real.m_eos.m_constants.perturb = h;
      elems_h.m_state.import_values(elems_ref.m_state,np1);
      dirk_real.run_initial_guess(np1,elems_h,hvcoord);

      // Compute error on initial guess deriv
      auto phih = ekat::scalarize(elems_h.m_derived.m_divdp_proj);

      auto linf = KOKKOS_LAMBDA(int ie, int ip, int jp, int ilev, Real& accum) {
        Real dphi_fd = (phih(ie,ip,jp,ilev) - phi0(ie,ip,jp,ilev)) / h;
        Real dphi_ex = dphi(ie,ip,jp,ilev).fastAccessDx(0);
        Real lcl_err = Kokkos::abs(dphi_fd-dphi_ex);
        if (lcl_err > accum) accum = lcl_err;
      };
      Kokkos::parallel_reduce(policy,linf,Kokkos::Max<Real>(eh.emplace_back(0)));
    }
    std::cout << std::setprecision(16) << "      h = [" << ekat::join(h_vals,",") << "]\n";
    std::cout << std::setprecision(16) << "      |dv/dp - FD(dvdp)|_inf = [" << ekat::join(eh,",") << "]\n";
    const Real min_err = *std::min_element(eh.begin(),eh.end());
    REQUIRE ((eh[0]<atol or min_err<eh[0]*rtol));
  }

  SECTION ("run") {
    const bool bfb_solver = false;
    Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>> policy({0,0,0,0},{num_elems,np,np,nlev+1});
    for (Real alphadt_nm1 : {0.0, 0.3}) {
      nm1 = alphadt_nm1 == 0.0 ? -1 : 0;
      printf("-> alphadt_nm1: %f\n", alphadt_nm1);
      for (Real alphadt_n0 : {0.0, 0.7}) {
        printf("  -> alphadt_n0: %f\n", alphadt_n0);

        // Run dirk_dp
        elems_dp.m_state.import_values(elems_ref.m_state,0);
        elems_dp.m_state.import_values(elems_ref.m_state,1);
        elems_dp.m_state.import_values(elems_ref.m_state,2);
        dirk_dp.run(nm1,alphadt_nm1,n0,alphadt_n0,np1,dt2,elems_dp,hvcoord,bfb_solver);
        auto dphi = ekat::scalarize(elems_dp.m_state.m_phinh_i);

        // Run dirk_real with no perturbation
        dirk_real.m_eos.m_constants.perturb = 0;
        elems_h0.m_state.import_values(elems_ref.m_state,0);
        elems_h0.m_state.import_values(elems_ref.m_state,1);
        elems_h0.m_state.import_values(elems_ref.m_state,2);
        dirk_real.run(nm1,alphadt_nm1,n0,alphadt_n0,np1,dt2,elems_h0,hvcoord,bfb_solver);
        auto phi0 = ekat::scalarize(elems_h0.m_state.m_phinh_i);

        // Run perturbed dirk_real
        std::vector<Real> eh;
        for (auto h : h_vals) {
          dirk_real.m_eos.m_constants.perturb = h;
          elems_h.m_state.import_values(elems_ref.m_state,0);
          elems_h.m_state.import_values(elems_ref.m_state,1);
          elems_h.m_state.import_values(elems_ref.m_state,2);
          dirk_real.run(nm1,alphadt_nm1,n0,alphadt_n0,np1,dt2,elems_h,hvcoord,bfb_solver);
          auto phih = ekat::scalarize(elems_h.m_state.m_phinh_i);

          // Compute error on phi(np1)
          auto linf = KOKKOS_LAMBDA(int ie, int ip, int jp, int ilev, Real& accum) {
            Real dphi_fd = (phih(ie,np1,ip,jp,ilev) - phi0(ie,np1,ip,jp,ilev)) / h;
            Real dphi_ex = dphi(ie,np1,ip,jp,ilev).fastAccessDx(0);
            Real lcl_err = Kokkos::abs(dphi_fd-dphi_ex);
            if (lcl_err > accum) accum = lcl_err;
          };
          Kokkos::parallel_reduce(policy,linf,Kokkos::Max<Real>(eh.emplace_back(0)));
        }
        std::cout << std::setprecision(16) << "      h = [" << ekat::join(h_vals,",") << "]\n";
        std::cout << std::setprecision(16) << "      |dv/dp - FD(dvdp)|_inf = [" << ekat::join(eh,",") << "]\n";
        const Real min_err = *std::min_element(eh.begin(),eh.end());
        REQUIRE ((eh[0]<atol or min_err<eh[0]*rtol));
      }
    }
  }

  Context::finalize_singleton();
}
