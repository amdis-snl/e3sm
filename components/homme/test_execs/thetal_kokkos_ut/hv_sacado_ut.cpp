#include <catch2/catch.hpp>

#include <random>
#include <iomanip>

#include "Types.hpp"
#include "Context.hpp"
#include "Elements.hpp"
#include "HyperviscosityFunctorImpl.hpp"
#include "HybridVCoord.hpp"
#include "ReferenceElement.hpp"
#include "SimulationParams.hpp"
#include "SphereOperators.hpp"
#include "mpi/Connectivity.hpp"
#include "mpi/MpiBuffersManager.hpp"

#include "utilities/TestUtils.hpp"

#include <ekat_string_utils.hpp>

using namespace Homme;

namespace {
constexpr int last_interface_lev_idx = NUM_INTERFACE_LEV - 1;

template<typename ST>
void init_ref_and_derived (ElementsST<ST>& e) {
  Kokkos::deep_copy(e.m_state.m_ref_states.dp_ref, Real(0));
  Kokkos::deep_copy(e.m_state.m_ref_states.theta_ref, Real(0));
  Kokkos::deep_copy(e.m_state.m_ref_states.phi_i_ref, Real(0));
  Kokkos::deep_copy(e.m_derived.m_dpdiss_ave, ST(0));
  Kokkos::deep_copy(e.m_derived.m_dpdiss_biharmonic, ST(0));
}

void init_connectivity_and_buffers (Context& c, const int num_elems) {
  auto& conn = c.create<Connectivity>();
  conn.set_comm(c.get<ekat::Comm>());
  conn.set_num_elements(num_elems);
  conn.set_max_corner_elements(1);
  conn.finalize(false);

  auto& bmm = c.create<MpiBuffersManagerMap>();
  if (!bmm.is_connectivity_set()) {
    bmm.set_connectivity(c.get_ptr<Connectivity>());
  }
}

SimulationParams init_params () {
  SimulationParams params;
  params.params_set = true;
  params.theta_hydrostatic_mode = false;
  params.hypervis_subcycle = 1;
  params.hypervis_subcycle_tom = 0;
  params.hypervis_scaling = 0.0;
  params.nu = 1e-3;
  params.nu_div = 1e-3;
  params.nu_top = 0.0;
  params.nu_p = 0.0;
  params.nu_s = 1e-3;
  params.nu_ratio1 = 1.0;
  params.nu_ratio2 = 1.0;
  return params;
}

} // namespace

TEST_CASE ("hyperviscosity_dp_and_jv_testing")
{
  std::random_device rd;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");

  using rpdf = std::uniform_real_distribution<Real>;
  using rngAlg = std::mt19937_64;
  rngAlg engine(seed);

  Context::finalize_singleton();
  auto& c = Context::singleton();
  c.create<ekat::Comm>(MPI_COMM_WORLD);

  constexpr int num_elems = 1;
  const int np1 = 1;
  const Real dt = rpdf(1e-4,1e-2)(engine);
  const Real eta_ave_w = 1.0;
  const Real atol = 1e-6;
  const Real rtol = 1e-4;

  auto params = init_params();
  auto& hvcoord = c.create<HybridVCoord>();
  hvcoord.random_init(seed);
  c.create<SimulationParams>() = params;

  auto& ref_FE = c.create<ReferenceElement>();
  ref_FE.random_init(seed);

  init_connectivity_and_buffers(c, num_elems);

  ElementsST<Real> elems_ref, elems_0, elems_h, elems_jv;
  ElementsST<DpFadType> elems_dp;
  ElementsST<DxFadTypeHypervis> elems_dx;
  elems_ref.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  elems_0.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  elems_h.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  elems_jv.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  elems_dp.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  elems_dx.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);

  auto& geo = elems_ref.m_geometry = elems_0.m_geometry = elems_h.m_geometry
            = elems_jv.m_geometry = elems_dp.m_geometry = elems_dx.m_geometry;
  geo.randomize(seed);

  init_ref_and_derived(elems_ref);
  init_ref_and_derived(elems_0);
  init_ref_and_derived(elems_h);
  init_ref_and_derived(elems_jv);
  init_ref_and_derived(elems_dp);
  init_ref_and_derived(elems_dx);

  c.create<SphereOperatorsST<Real>>().setup(geo, ref_FE);
  c.create<SphereOperatorsST<DpFadType>>().setup(geo, ref_FE);
  c.create<SphereOperatorsST<DxFadTypeHypervis>>().setup(geo, ref_FE);

  const auto max_pressure = 1000.0 + hvcoord.ps0;
  elems_ref.m_state.randomize(seed,max_pressure,hvcoord.ps0,hvcoord.hybrid_ai0,geo.m_phis);
  elems_0.m_state.import_values(elems_ref.m_state,np1);
  elems_h.m_state.import_values(elems_ref.m_state,np1);
  elems_dp.m_state.import_values(elems_ref.m_state,np1);
  elems_dx.m_state.import_values(elems_ref.m_state,np1);

  ExecViewManaged<Real*[2][NP][NP][NUM_LEV]> Vv("", num_elems);
  ExecViewManaged<Real*[NP][NP][NUM_LEV]> Vdp("", num_elems);
  ExecViewManaged<Real*[NP][NP][NUM_LEV]> Vvth("", num_elems);
  ExecViewManaged<Real*[NP][NP][NUM_LEV_P]> Vw_pack("", num_elems);
  ExecViewManaged<Real*[NP][NP][NUM_LEV_P]> Vphi_pack("", num_elems);
  auto Vw = ekat::scalarize(Vw_pack);
  auto Vphi = ekat::scalarize(Vphi_pack);
  genRandArray(Vv, engine, rpdf(-1.0,1.0));
  genRandArray(Vdp, engine, rpdf(-1.0,1.0));
  genRandArray(Vvth, engine, rpdf(-1.0,1.0));
  genRandArray(Vw_pack, engine, rpdf(-1.0,1.0));
  genRandArray(Vphi_pack, engine, rpdf(-1.0,1.0));

  auto v_fad = ekat::scalarize(elems_dp.m_state.m_v);
  auto dp_fad = ekat::scalarize(elems_dp.m_state.m_dp3d);
  auto vth_fad = ekat::scalarize(elems_dp.m_state.m_vtheta_dp);
  auto w_fad = ekat::scalarize(elems_dp.m_state.m_w_i);
  auto phi_fad = ekat::scalarize(elems_dp.m_state.m_phinh_i);
  auto set_derivs = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
    v_fad  (ie,np1,0,ip,jp,k).fastAccessDx(0) = Vv  (ie,0,ip,jp,k);
    v_fad  (ie,np1,1,ip,jp,k).fastAccessDx(0) = Vv  (ie,1,ip,jp,k);
    dp_fad (ie,np1,ip,jp,k).fastAccessDx(0) = Vdp (ie,ip,jp,k);
    vth_fad(ie,np1,ip,jp,k).fastAccessDx(0) = Vvth(ie,ip,jp,k);
    w_fad  (ie,np1,ip,jp,k).fastAccessDx(0) = Vw  (ie,ip,jp,k);
    phi_fad(ie,np1,ip,jp,k).fastAccessDx(0) = Vphi(ie,ip,jp,k);
  };
  auto set_derivs_int = KOKKOS_LAMBDA (const int ie, const int ip, const int jp) {
    w_fad  (ie,np1,ip,jp,last_interface_lev_idx).fastAccessDx(0) = Vw  (ie,ip,jp,last_interface_lev_idx);
    phi_fad(ie,np1,ip,jp,last_interface_lev_idx).fastAccessDx(0) = Vphi(ie,ip,jp,last_interface_lev_idx);
  };
  Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>> p4({0,0,0,0},{num_elems,NP,NP,NUM_PHYSICAL_LEV});
  Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>> p3({0,0,0},{num_elems,NP,NP});
  Kokkos::parallel_for(p4, set_derivs);
  Kokkos::parallel_for(p3, set_derivs_int);

  HyperviscosityFunctorImplST<Real> hv_0(params,geo,elems_0.m_state,elems_0.m_derived);
  HyperviscosityFunctorImplST<Real> hv_h(params,geo,elems_h.m_state,elems_h.m_derived);
  HyperviscosityFunctorImplST<DpFadType> hv_dp(params,geo,elems_dp.m_state,elems_dp.m_derived);
  HyperviscosityFunctorImplST<DxFadTypeHypervis> hv_dx(params,geo,elems_dx.m_state,elems_dx.m_derived);

  FunctorsBuffersManager fbm;
  fbm.request_size(hv_0.requested_buffer_size());
  fbm.request_size(hv_h.requested_buffer_size());
  fbm.request_size(hv_dp.requested_buffer_size());
  fbm.request_size(hv_dx.requested_buffer_size());
  fbm.allocate();
  hv_0.init_buffers(fbm);
  hv_h.init_buffers(fbm);
  hv_dp.init_buffers(fbm);
  hv_dx.init_buffers(fbm);
  hv_0.init_boundary_exchanges();
  hv_h.init_boundary_exchanges();
  hv_dp.init_boundary_exchanges();
  hv_dx.init_boundary_exchanges();

  hv_dp.run(np1,dt,eta_ave_w);
  hv_0.run(np1,dt,eta_ave_w);

  auto v_0 = ekat::scalarize(elems_0.m_state.m_v);
  auto dp_0 = ekat::scalarize(elems_0.m_state.m_dp3d);
  auto vth_0 = ekat::scalarize(elems_0.m_state.m_vtheta_dp);
  auto w_0 = ekat::scalarize(elems_0.m_state.m_w_i);
  auto phi_0 = ekat::scalarize(elems_0.m_state.m_phinh_i);
  auto v_dp = ekat::scalarize(elems_dp.m_state.m_v);
  auto dp_dp = ekat::scalarize(elems_dp.m_state.m_dp3d);
  auto vth_dp = ekat::scalarize(elems_dp.m_state.m_vtheta_dp);
  auto w_dp = ekat::scalarize(elems_dp.m_state.m_w_i);
  auto phi_dp = ekat::scalarize(elems_dp.m_state.m_phinh_i);

  const std::vector<Real> h_vals = {1e-1,1e-2,1e-3,1e-4,1e-5,1e-6};
  std::vector<Real> err_fd;
  for (const auto h : h_vals) {
    elems_h.m_state.import_values(elems_ref.m_state,np1);
    auto v_h = ekat::scalarize(elems_h.m_state.m_v);
    auto dp_h = ekat::scalarize(elems_h.m_state.m_dp3d);
    auto vth_h = ekat::scalarize(elems_h.m_state.m_vtheta_dp);
    auto w_h = ekat::scalarize(elems_h.m_state.m_w_i);
    auto phi_h = ekat::scalarize(elems_h.m_state.m_phinh_i);

    auto perturb = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
      v_h   (ie,np1,0,ip,jp,k) += h*Vv  (ie,0,ip,jp,k);
      v_h   (ie,np1,1,ip,jp,k) += h*Vv  (ie,1,ip,jp,k);
      dp_h (ie,np1,ip,jp,k) += h*Vdp (ie,ip,jp,k);
      vth_h(ie,np1,ip,jp,k) += h*Vvth(ie,ip,jp,k);
      w_h  (ie,np1,ip,jp,k) += h*Vw  (ie,ip,jp,k);
      phi_h(ie,np1,ip,jp,k) += h*Vphi(ie,ip,jp,k);
    };
    auto perturb_int = KOKKOS_LAMBDA (const int ie, const int ip, const int jp) {
      w_h  (ie,np1,ip,jp,last_interface_lev_idx) += h*Vw  (ie,ip,jp,last_interface_lev_idx);
      phi_h(ie,np1,ip,jp,last_interface_lev_idx) += h*Vphi(ie,ip,jp,last_interface_lev_idx);
    };
    Kokkos::parallel_for(p4, perturb);
    Kokkos::parallel_for(p3, perturb_int);

    hv_h.run(np1,dt,eta_ave_w);

    auto v_out_h = ekat::scalarize(elems_h.m_state.m_v);
    auto dp_out_h = ekat::scalarize(elems_h.m_state.m_dp3d);
    auto vth_out_h = ekat::scalarize(elems_h.m_state.m_vtheta_dp);
    auto w_out_h = ekat::scalarize(elems_h.m_state.m_w_i);
    auto phi_out_h = ekat::scalarize(elems_h.m_state.m_phinh_i);
    auto linf = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k, Real& accum) {
      const Real du_fd   = (v_out_h(ie,np1,0,ip,jp,k) - v_0(ie,np1,0,ip,jp,k))/h;
      const Real dv_fd   = (v_out_h(ie,np1,1,ip,jp,k) - v_0(ie,np1,1,ip,jp,k))/h;
      const Real ddp_fd  = (dp_out_h(ie,np1,ip,jp,k)  - dp_0 (ie,np1,ip,jp,k))/h;
      const Real dvth_fd = (vth_out_h(ie,np1,ip,jp,k) - vth_0(ie,np1,ip,jp,k))/h;
      const Real dw_fd   = (w_out_h(ie,np1,ip,jp,k)   - w_0(ie,np1,ip,jp,k))/h;
      const Real dphi_fd = (phi_out_h(ie,np1,ip,jp,k) - phi_0(ie,np1,ip,jp,k))/h;
      const Real du_ex   = v_dp(ie,np1,0,ip,jp,k).fastAccessDx(0);
      const Real dv_ex   = v_dp(ie,np1,1,ip,jp,k).fastAccessDx(0);
      const Real ddp_ex  = dp_dp (ie,np1,ip,jp,k).fastAccessDx(0);
      const Real dvth_ex = vth_dp(ie,np1,ip,jp,k).fastAccessDx(0);
      const Real dw_ex   = w_dp(ie,np1,ip,jp,k).fastAccessDx(0);
      const Real dphi_ex = phi_dp(ie,np1,ip,jp,k).fastAccessDx(0);
      Real lcl = Kokkos::abs(du_fd-du_ex);
      lcl = Kokkos::max(lcl, Kokkos::abs(dv_fd-dv_ex));
      lcl = Kokkos::max(lcl, Kokkos::abs(ddp_fd-ddp_ex));
      lcl = Kokkos::max(lcl, Kokkos::abs(dvth_fd-dvth_ex));
      lcl = Kokkos::max(lcl, Kokkos::abs(dw_fd-dw_ex));
      lcl = Kokkos::max(lcl, Kokkos::abs(dphi_fd-dphi_ex));
      if (lcl > accum) accum = lcl;
    };
    auto linf_int = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, Real& accum) {
      const Real dwl_fd   = (w_out_h(ie,np1,ip,jp,last_interface_lev_idx)   - w_0(ie,np1,ip,jp,last_interface_lev_idx))/h;
      const Real dphil_fd = (phi_out_h(ie,np1,ip,jp,last_interface_lev_idx) - phi_0(ie,np1,ip,jp,last_interface_lev_idx))/h;
      const Real dwl_ex   = w_dp(ie,np1,ip,jp,last_interface_lev_idx).fastAccessDx(0);
      const Real dphil_ex = phi_dp(ie,np1,ip,jp,last_interface_lev_idx).fastAccessDx(0);
      Real lcl = Kokkos::abs(dwl_fd-dwl_ex);
      lcl = Kokkos::max(lcl, Kokkos::abs(dphil_fd-dphil_ex));
      if (lcl > accum) accum = lcl;
    };
    Real err_phys = 0;
    Real err_int  = 0;
    Kokkos::parallel_reduce(p4, linf, Kokkos::Max<Real>(err_phys));
    Kokkos::parallel_reduce(p3, linf_int, Kokkos::Max<Real>(err_int));
    err_fd.emplace_back(Kokkos::max(err_phys,err_int));
  }

  std::cout << std::setprecision(16)
            << "  h = [" << ekat::join(h_vals,",") << "]\n"
            << "  |dF/dp - FD|_inf = [" << ekat::join(err_fd,",") << "]\n";
  const Real min_err = *std::min_element(err_fd.begin(),err_fd.end());
  REQUIRE((err_fd[0] < atol or min_err < err_fd[0]*rtol));

  elems_dx.m_state.import_values(elems_ref.m_state,np1);
  elems_jv.m_state.import_values_from_deriv(elems_dp.m_state,np1,0);

  hv_dx.init_JV(np1,elems_dx);
  hv_dx.run(np1,dt,eta_ave_w);
  hv_dx.run_JV(np1,elems_dx,elems_jv.m_state);

  auto v_jv = ekat::scalarize(elems_jv.m_state.m_v);
  auto dp_jv = ekat::scalarize(elems_jv.m_state.m_dp3d);
  auto vth_jv = ekat::scalarize(elems_jv.m_state.m_vtheta_dp);
  auto w_jv = ekat::scalarize(elems_jv.m_state.m_w_i);
  auto phi_jv = ekat::scalarize(elems_jv.m_state.m_phinh_i);

  auto v_dp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v_dp);
  auto dp_dp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dp_dp);
  auto vth_dp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vth_dp);
  auto w_dp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w_dp);
  auto phi_dp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), phi_dp);
  auto v_jv_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v_jv);
  auto dp_jv_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dp_jv);
  auto vth_jv_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vth_jv);
  auto w_jv_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w_jv);
  auto phi_jv_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), phi_jv);

  for (int ie = 0; ie < num_elems; ++ie) {
    for (int ip = 0; ip < NP; ++ip) {
      for (int jp = 0; jp < NP; ++jp) {
        for (int k = 0; k < NUM_PHYSICAL_LEV; ++k) {
          const auto du_src = v_jv_h(ie,np1,0,ip,jp,k);
          const auto du_tgt = v_dp_h(ie,np1,0,ip,jp,k).dx(0);
          CHECK_THAT(du_src, Catch::WithinRel(du_tgt, rtol) || Catch::WithinAbs(du_tgt, atol));

          const auto dv_src = v_jv_h(ie,np1,1,ip,jp,k);
          const auto dv_tgt = v_dp_h(ie,np1,1,ip,jp,k).dx(0);
          CHECK_THAT(dv_src, Catch::WithinRel(dv_tgt, rtol) || Catch::WithinAbs(dv_tgt, atol));

          const auto ddp_src = dp_jv_h(ie,np1,ip,jp,k);
          const auto ddp_tgt = dp_dp_h(ie,np1,ip,jp,k).dx(0);
          CHECK_THAT(ddp_src, Catch::WithinRel(ddp_tgt, rtol) || Catch::WithinAbs(ddp_tgt, atol));

          const auto dvth_src = vth_jv_h(ie,np1,ip,jp,k);
          const auto dvth_tgt = vth_dp_h(ie,np1,ip,jp,k).dx(0);
          CHECK_THAT(dvth_src, Catch::WithinRel(dvth_tgt, rtol) || Catch::WithinAbs(dvth_tgt, atol));

          const auto dw_src = w_jv_h(ie,np1,ip,jp,k);
          const auto dw_tgt = w_dp_h(ie,np1,ip,jp,k).dx(0);
          CHECK_THAT(dw_src, Catch::WithinRel(dw_tgt, rtol) || Catch::WithinAbs(dw_tgt, atol));

          const auto dphi_src = phi_jv_h(ie,np1,ip,jp,k);
          const auto dphi_tgt = phi_dp_h(ie,np1,ip,jp,k).dx(0);
          CHECK_THAT(dphi_src, Catch::WithinRel(dphi_tgt, rtol) || Catch::WithinAbs(dphi_tgt, atol));
        }
        const auto dwl_src = w_jv_h(ie,np1,ip,jp,last_interface_lev_idx);
        const auto dwl_tgt = w_dp_h(ie,np1,ip,jp,last_interface_lev_idx).dx(0);
        CHECK_THAT(dwl_src, Catch::WithinRel(dwl_tgt, rtol) || Catch::WithinAbs(dwl_tgt, atol));
        const auto dphil_src = phi_jv_h(ie,np1,ip,jp,last_interface_lev_idx);
        const auto dphil_tgt = phi_dp_h(ie,np1,ip,jp,last_interface_lev_idx).dx(0);
        CHECK_THAT(dphil_src, Catch::WithinRel(dphil_tgt, rtol) || Catch::WithinAbs(dphil_tgt, atol));
      }
    }
  }

  Context::finalize_singleton();
}

TEST_CASE ("hyperviscosity_jtv_testing") {
  std::random_device rd;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");

  using rpdf = std::uniform_real_distribution<Real>;
  using rngAlg = std::mt19937_64;
  rngAlg engine(seed);

  Context::finalize_singleton();
  auto& c = Context::singleton();
  c.create<ekat::Comm>(MPI_COMM_WORLD);

  constexpr int num_elems = 1;
  const int np1 = 1;
  const Real dt = rpdf(1e-4,1e-2)(engine);
  const Real eta_ave_w = 1.0;
  const Real atol = 1e-8;
  const Real rtol = 1e-8;

  auto params = init_params();
  auto& hvcoord = c.create<HybridVCoord>();
  hvcoord.random_init(seed);
  c.create<SimulationParams>() = params;

  auto& ref_FE = c.create<ReferenceElement>();
  ref_FE.random_init(seed);

  init_connectivity_and_buffers(c, num_elems);

  ElementsST<DxFadTypeHypervis> elems_dx;
  elems_dx.init(num_elems,false,true,PhysicalConstants::rearth0,-1,true);
  elems_dx.m_geometry.randomize(seed);
  init_ref_and_derived(elems_dx);

  ElementsStateST<Real> adj_a, adj_b, in_a, in_b;
  adj_a.init(num_elems);
  adj_b.init(num_elems);
  in_a.init(num_elems);
  in_b.init(num_elems);

  c.create<SphereOperatorsST<DxFadTypeHypervis>>().setup(elems_dx.m_geometry, ref_FE);

  const auto max_pressure = 1000.0 + hvcoord.ps0;
  elems_dx.m_state.randomize(seed,max_pressure,hvcoord.ps0,hvcoord.hybrid_ai0,elems_dx.m_geometry.m_phis);

  adj_a.randomize(seed,max_pressure,hvcoord.ps0,hvcoord.hybrid_ai0,elems_dx.m_geometry.m_phis);
  adj_b.randomize(seed+1,max_pressure,hvcoord.ps0,hvcoord.hybrid_ai0,elems_dx.m_geometry.m_phis);
  in_a.import_values(adj_a,np1);
  in_b.import_values(adj_b,np1);

  HyperviscosityFunctorImplST<DxFadTypeHypervis> hv_dx(params,elems_dx.m_geometry,elems_dx.m_state,elems_dx.m_derived);
  FunctorsBuffersManager fbm;
  fbm.request_size(hv_dx.requested_buffer_size());
  fbm.allocate();
  hv_dx.init_buffers(fbm);
  hv_dx.init_boundary_exchanges();

  hv_dx.init_JV(np1,elems_dx);
  hv_dx.run(np1,dt,eta_ave_w);

  hv_dx.run_JV(np1,elems_dx,adj_b);
  hv_dx.run_JtV(np1,elems_dx,adj_a);

  auto ain_v = ekat::scalarize(in_a.m_v);
  auto ain_dp = ekat::scalarize(in_a.m_dp3d);
  auto ain_vth = ekat::scalarize(in_a.m_vtheta_dp);
  auto ain_w = ekat::scalarize(in_a.m_w_i);
  auto ain_phi = ekat::scalarize(in_a.m_phinh_i);
  auto bin_v = ekat::scalarize(in_b.m_v);
  auto bin_dp = ekat::scalarize(in_b.m_dp3d);
  auto bin_vth = ekat::scalarize(in_b.m_vtheta_dp);
  auto bin_w = ekat::scalarize(in_b.m_w_i);
  auto bin_phi = ekat::scalarize(in_b.m_phinh_i);
  auto aout_v = ekat::scalarize(adj_a.m_v);
  auto aout_dp = ekat::scalarize(adj_a.m_dp3d);
  auto aout_vth = ekat::scalarize(adj_a.m_vtheta_dp);
  auto aout_w = ekat::scalarize(adj_a.m_w_i);
  auto aout_phi = ekat::scalarize(adj_a.m_phinh_i);
  auto bout_v = ekat::scalarize(adj_b.m_v);
  auto bout_dp = ekat::scalarize(adj_b.m_dp3d);
  auto bout_vth = ekat::scalarize(adj_b.m_vtheta_dp);
  auto bout_w = ekat::scalarize(adj_b.m_w_i);
  auto bout_phi = ekat::scalarize(adj_b.m_phinh_i);

  Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>> p4({0,0,0,0},{num_elems,NP,NP,NUM_PHYSICAL_LEV});
  Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>> p3({0,0,0},{num_elems,NP,NP});
  Real2 dot;
  Real2 dot_int;
  Kokkos::parallel_reduce(p4, KOKKOS_LAMBDA(const int ie, const int ip, const int jp, const int k, Real2& acc) {
    acc.v[0] += ain_v(ie,np1,0,ip,jp,k) * bout_v(ie,np1,0,ip,jp,k)
             +  ain_v(ie,np1,1,ip,jp,k) * bout_v(ie,np1,1,ip,jp,k)
             +  ain_dp (ie,np1,ip,jp,k) * bout_dp (ie,np1,ip,jp,k)
             +  ain_vth(ie,np1,ip,jp,k) * bout_vth(ie,np1,ip,jp,k)
             +  ain_w  (ie,np1,ip,jp,k) * bout_w  (ie,np1,ip,jp,k)
             +  ain_phi(ie,np1,ip,jp,k) * bout_phi(ie,np1,ip,jp,k);
    acc.v[1] += aout_v(ie,np1,0,ip,jp,k) * bin_v(ie,np1,0,ip,jp,k)
             +  aout_v(ie,np1,1,ip,jp,k) * bin_v(ie,np1,1,ip,jp,k)
             +  aout_dp (ie,np1,ip,jp,k) * bin_dp (ie,np1,ip,jp,k)
             +  aout_vth(ie,np1,ip,jp,k) * bin_vth(ie,np1,ip,jp,k)
             +  aout_w  (ie,np1,ip,jp,k) * bin_w  (ie,np1,ip,jp,k)
             +  aout_phi(ie,np1,ip,jp,k) * bin_phi(ie,np1,ip,jp,k);
  }, dot);
  Kokkos::parallel_reduce(p3, KOKKOS_LAMBDA(const int ie, const int ip, const int jp, Real2& acc) {
    acc.v[0] += ain_w  (ie,np1,ip,jp,last_interface_lev_idx) * bout_w  (ie,np1,ip,jp,last_interface_lev_idx)
             +  ain_phi(ie,np1,ip,jp,last_interface_lev_idx) * bout_phi(ie,np1,ip,jp,last_interface_lev_idx);
    acc.v[1] += aout_w  (ie,np1,ip,jp,last_interface_lev_idx) * bin_w  (ie,np1,ip,jp,last_interface_lev_idx)
             +  aout_phi(ie,np1,ip,jp,last_interface_lev_idx) * bin_phi(ie,np1,ip,jp,last_interface_lev_idx);
  }, dot_int);
  dot += dot_int;

  CHECK_THAT(dot.v[0], Catch::WithinRel(dot.v[1], rtol) || Catch::WithinAbs(dot.v[1], atol));

  Context::finalize_singleton();
}
