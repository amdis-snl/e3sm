/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_HYPERVISCOSITY_FUNCTOR_IMPL_HPP
#define HOMMEXX_HYPERVISCOSITY_FUNCTOR_IMPL_HPP

#include "Elements.hpp"
#include "ColumnOps.hpp"
#include "EquationOfState.hpp"
#include "ElementOps.hpp"
#include "FunctorsBuffersManager.hpp"
#include "HybridVCoord.hpp"
#include "KernelVariables.hpp"
#include "SimulationParams.hpp"
#include "SphereOperators.hpp"

#include <memory>

#include "profiling.hpp"

namespace Homme
{

template<typename ST>
class BoundaryExchangeST;
struct FunctorsBuffersManager;

template<typename ST>
class HyperviscosityFunctorImplST
{
  using PT = PackType<ST>;

  // TODO: don't pass nu_ratio1/2. Instead, do like in F90: compute them from
  //       nu, nu_div, and hv_scaling
  struct HyperviscosityData {
    HyperviscosityData(const int hypervis_subcycle_in, 
                       const int hypervis_subcycle_tom_in, 
                       const Real nu_ratio1_in, const Real nu_ratio2_in, const Real nu_top_in,
                       const Real nu_in, const Real nu_p_in, const Real nu_s_in,
                       const Real hypervis_scaling_in)
                      : hypervis_subcycle(hypervis_subcycle_in) 
                      , hypervis_subcycle_tom(hypervis_subcycle_tom_in)
                      , nu_ratio1(nu_ratio1_in), nu_ratio2(nu_ratio2_in)
                      , nu_top(nu_top_in), nu(nu_in), nu_p(nu_p_in), nu_s(nu_s_in)
                      , consthv(hypervis_scaling_in == 0) {}

    const int   hypervis_subcycle;
    const int   hypervis_subcycle_tom;

    Real  nu_ratio1;
    Real  nu_ratio2;

#ifdef HOMMEXX_TEST_HYPERVIS_FAD
    ST rel_perturb;
    ST nu;
#else
    const Real  nu;
#endif
    const Real  nu_top;
    const Real  nu_p;
    const Real  nu_s;

    int         np1; // The time-level on which to apply hv
    Real        dt;
    Real        dt_hvs;
    Real        dt_hvs_tom;

    Real        eta_ave_w;

    bool consthv;
  };//hyperviscosityData

  struct Buffers {
    ExecViewManaged<PT * [NP][NP][NUM_LEV]>    dptens;
    ExecViewManaged<PT * [NP][NP][NUM_LEV]>    ttens;
    ExecViewManaged<PT * [NP][NP][NUM_LEV]>    wtens;
    ExecViewManaged<PT * [NP][NP][NUM_LEV]>    phitens;
    ExecViewManaged<PT * [2][NP][NP][NUM_LEV]> vtens;
  };//buffers

public:

  struct TagFirstLaplaceHV {};
  struct TagSecondLaplaceConstHV {};
  struct TagSecondLaplaceTensorHV {};
  struct TagUpdateStates {};
  struct TagApplyInvMass {};
  struct TagHyperPreExchange {};
  struct TagNutopUpdateStates {};
  struct TagNutopLaplace {};

  HyperviscosityFunctorImplST (const SimulationParams&           params,
                               const ElementsGeometry&           geometry,
                               const ElementsStateST<ST>&        state,
                               const ElementsDerivedStateST<ST>& derived);

  HyperviscosityFunctorImplST (const int num_elems, const SimulationParams& params);

  void init_params(const SimulationParams& params);

  void setup(const ElementsGeometry&           geometry,
             const ElementsStateST<ST>&        state,
             const ElementsDerivedStateST<ST>& derived);

  int requested_buffer_size () const;
  void init_buffers (const FunctorsBuffersManager& fbm);
  void init_boundary_exchanges();

  void run (const int np1, const Real dt, const Real eta_ave_w);

  void biharmonic_wk_theta () const;

#ifdef HOMMEXX_ENABLE_FAD_TYPES
  // Hyperviscosity Dx FAD layout (2*NP*NP derivatives per level).
  // The Jacobian is block-diagonal; slots are reused across decoupled groups:
  //   Slots 0..NP*NP-1:       u (Group A), vtheta_dp (Group B), w_i (Group C), phinh_i (Group D)
  //   Slots NP*NP..2*NP*NP-1: v (Group A), dp3d      (Group B)
  // At the surface interface level, run_w_surf overwrites w_i with f(u_surf,v_surf),
  // so its slots carry d(w_surf)/d(u) and d(w_surf)/d(v) instead of d/d(w_init).
  // run_JV / run_JtV apply ad-hoc per-variable logic to interpret slots correctly.
  static constexpr int fad_offset_u   = 0;
  static constexpr int fad_offset_v   = NP*NP;
  static constexpr int fad_offset_vth = 0;      // shared with u (Groups A,B decoupled)
  static constexpr int fad_offset_dp  = NP*NP;  // shared with v
  static constexpr int fad_offset_w   = 0;      // shared with u (Group C decoupled)
  static constexpr int fad_offset_phi = 0;      // shared with u (Group D decoupled)

  template<typename MyST = ST>
  std::enable_if_t<not std::is_same_v<MyST, DxFadTypeHypervis>>
  init_JV (const int, const ElementsST<ST>&) = delete;

  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, DxFadTypeHypervis>>
  init_JV (const int np1, const ElementsST<ST>& e) {
    using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
    const int nelem = e.m_state.num_elems();
    auto p4_mid = md_range_t({0,0,0,0}, {nelem,NP,NP,NUM_PHYSICAL_LEV});
    auto p4_int = md_range_t({0,0,0,0}, {nelem,NP,NP,NUM_INTERFACE_LEV});

    auto dvdx_v   = ekat::scalarize(e.m_state.m_v);
    auto ddpdx_v  = ekat::scalarize(e.m_state.m_dp3d);
    auto dvthdx_v = ekat::scalarize(e.m_state.m_vtheta_dp);
    auto dwdx_v   = ekat::scalarize(e.m_state.m_w_i);
    auto dphidx_v = ekat::scalarize(e.m_state.m_phinh_i);

    const int offset_u   = fad_offset_u;
    const int offset_v   = fad_offset_v;
    const int offset_dp  = fad_offset_dp;
    const int offset_vth = fad_offset_vth;
    const int offset_w   = fad_offset_w;
    const int offset_phi = fad_offset_phi;

    auto init_dx_mid = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
      const int gp_idx = ip*NP + jp;
      auto& dudx   = dvdx_v  (ie,np1,0,ip,jp,k);
      auto& dvdx   = dvdx_v  (ie,np1,1,ip,jp,k);
      auto& ddpdx  = ddpdx_v (ie,np1,ip,jp,k);
      auto& dvthdx = dvthdx_v(ie,np1,ip,jp,k);
      dudx.zero();
      dvdx.zero();
      ddpdx.zero();
      dvthdx.zero();
      dudx.fastAccessDx  (offset_u   + gp_idx) = 1;
      dvdx.fastAccessDx  (offset_v   + gp_idx) = 1;
      ddpdx.fastAccessDx (offset_dp  + gp_idx) = 1;
      dvthdx.fastAccessDx(offset_vth + gp_idx) = 1;
    };
    auto init_dx_int = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
      const int gp_idx = ip*NP + jp;
      auto& dwdx   = dwdx_v  (ie,np1,ip,jp,k);
      auto& dphidx = dphidx_v(ie,np1,ip,jp,k);
      dwdx.zero();
      dphidx.zero();
      dwdx.fastAccessDx  (offset_w   + gp_idx) = 1;
      dphidx.fastAccessDx(offset_phi + gp_idx) = 1;
    };
    Kokkos::parallel_for(p4_mid, init_dx_mid);
    Kokkos::parallel_for(p4_int, init_dx_int);
  }

  template<typename MyST = ST>
  std::enable_if_t<not std::is_same_v<MyST, DxFadTypeHypervis>>
  run_JV (const int, const ElementsST<ST>&, ElementsStateST<Real>&) = delete;

  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, DxFadTypeHypervis>>
  run_JV (const int np1, const ElementsST<ST>& e, ElementsStateST<Real>& adj_state)
  {
    const int nelem = e.m_state.num_elems();

    auto dV_v   = ekat::scalarize(e.m_state.m_v);
    auto ddp_v  = ekat::scalarize(e.m_state.m_dp3d);
    auto dvth_v = ekat::scalarize(e.m_state.m_vtheta_dp);
    auto dw_v   = ekat::scalarize(e.m_state.m_w_i);
    auto dphi_v = ekat::scalarize(e.m_state.m_phinh_i);

    auto l_V    = ekat::scalarize(adj_state.m_v);
    auto l_dp   = ekat::scalarize(adj_state.m_dp3d);
    auto l_vth  = ekat::scalarize(adj_state.m_vtheta_dp);
    auto l_w    = ekat::scalarize(adj_state.m_w_i);
    auto l_phi  = ekat::scalarize(adj_state.m_phinh_i);

    ExecViewManaged<Real*[2][NP][NP][NUM_LEV]> out_V("", nelem);
    ExecViewManaged<Real*[NP][NP][NUM_LEV]> out_dp("", nelem);
    ExecViewManaged<Real*[NP][NP][NUM_LEV]> out_vth("", nelem);
    ExecViewManaged<Real*[NP][NP][NUM_LEV_P]> out_w_pack("", nelem);
    ExecViewManaged<Real*[NP][NP][NUM_LEV_P]> out_phi_pack("", nelem);
    auto out_w = ekat::scalarize(out_w_pack);
    auto out_phi = ekat::scalarize(out_phi_pack);

    using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
    auto p4_mid = md_range_t({0,0,0,0}, {nelem,NP,NP,NUM_PHYSICAL_LEV});
    auto p4_int = md_range_t({0,0,0,0}, {nelem,NP,NP,NUM_INTERFACE_LEV});

    // JV product rule for midpoint variables.
    // Block-diagonal structure: Group A {u,v} and Group B {vth,dp} are decoupled.
    // Slot gp_idx        = d/du[gp]  for u/v outputs = d/dvth[gp]  for vth/dp outputs
    // Slot NP*NP+gp_idx  = d/dv[gp]  for u/v outputs = d/ddp[gp]   for vth/dp outputs
    auto prod_rule_mid = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
      const auto& Ju   = dV_v  (ie,np1,0,ip,jp,k).dx();
      const auto& Jv   = dV_v  (ie,np1,1,ip,jp,k).dx();
      const auto& Jdp  = ddp_v (ie,np1,ip,jp,k).dx();
      const auto& Jvth = dvth_v(ie,np1,ip,jp,k).dx();

      Real l_u_new   = 0;
      Real l_v_new   = 0;
      Real l_dp_new  = 0;
      Real l_vth_new = 0;
      for (int m = 0; m < NP; ++m) {
        for (int n = 0; n < NP; ++n) {
          const int gp_idx = m*NP + n;
          // Group A {u,v}: slot gp_idx = d/du[gp], slot NP*NP+gp_idx = d/dv[gp]
          l_u_new   += Ju [gp_idx      ] * l_V  (ie,np1,0,m,n,k)
                    +  Ju [NP*NP+gp_idx] * l_V  (ie,np1,1,m,n,k);
          l_v_new   += Jv [gp_idx      ] * l_V  (ie,np1,0,m,n,k)
                    +  Jv [NP*NP+gp_idx] * l_V  (ie,np1,1,m,n,k);
          // Group B {vth,dp}: slot gp_idx = d/dvth[gp], slot NP*NP+gp_idx = d/ddp[gp]
          l_vth_new += Jvth[gp_idx      ] * l_vth(ie,np1,m,n,k)
                    +  Jvth[NP*NP+gp_idx] * l_dp (ie,np1,m,n,k);
          l_dp_new  += Jdp [gp_idx      ] * l_vth(ie,np1,m,n,k)
                    +  Jdp [NP*NP+gp_idx] * l_dp (ie,np1,m,n,k);
        }
      }
      out_V  (ie,0,ip,jp,k) = l_u_new;
      out_V  (ie,1,ip,jp,k) = l_v_new;
      out_dp (ie,ip,jp,k) = l_dp_new;
      out_vth(ie,ip,jp,k) = l_vth_new;
    };

    // JV product rule for interface variables.
    // Group C {w_i}: at interior levels slot gp_idx = d/dw[gp].
    //   At the surface level, run_w_surf overwrites w with f(u,v), so
    //   slot gp_idx = d/du[gp] and slot NP*NP+gp_idx = d/dv[gp].
    // Group D {phi}: slot gp_idx = d/dphi[gp] at all interface levels.
    auto prod_rule_int = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
      const auto& Jw   = dw_v  (ie,np1,ip,jp,k).dx();
      const auto& Jphi = dphi_v(ie,np1,ip,jp,k).dx();
      constexpr int last_physical_lev_idx = NUM_PHYSICAL_LEV-1;
      const int km = k < NUM_PHYSICAL_LEV ? k : last_physical_lev_idx;
      const bool is_surf = (k >= NUM_PHYSICAL_LEV);

      Real l_w_new   = 0;
      Real l_phi_new = 0;
      for (int m = 0; m < NP; ++m) {
        for (int n = 0; n < NP; ++n) {
          const int gp_idx = m*NP + n;
          if (!is_surf) {
            // Interior w (Group C): slot gp_idx = d(w)/d(w_init[gp])
            l_w_new += Jw[gp_idx] * l_w(ie,np1,m,n,k);
          } else {
            // Surface w: slots carry d(w_surf)/d(u[gp]) and d(w_surf)/d(v[gp])
            l_w_new += Jw[gp_idx      ] * l_V(ie,np1,0,m,n,km)
                    +  Jw[NP*NP+gp_idx] * l_V(ie,np1,1,m,n,km);
          }
          // phi (Group D): slot gp_idx = d(phi)/d(phi_init[gp]) at all levels
          l_phi_new += Jphi[gp_idx] * l_phi(ie,np1,m,n,k);
        }
      }
      out_w  (ie,ip,jp,k) = l_w_new;
      out_phi(ie,ip,jp,k) = l_phi_new;
    };

    auto copy_back_mid = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
      l_V  (ie,np1,0,ip,jp,k) = out_V  (ie,0,ip,jp,k);
      l_V  (ie,np1,1,ip,jp,k) = out_V  (ie,1,ip,jp,k);
      l_dp (ie,np1,ip,jp,k) = out_dp (ie,ip,jp,k);
      l_vth(ie,np1,ip,jp,k) = out_vth(ie,ip,jp,k);
    };

    auto copy_back_int = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
      l_w  (ie,np1,ip,jp,k) = out_w  (ie,ip,jp,k);
      l_phi(ie,np1,ip,jp,k) = out_phi(ie,ip,jp,k);
    };

    Kokkos::parallel_for(p4_mid, prod_rule_mid);
    Kokkos::parallel_for(p4_int, prod_rule_int);
    Kokkos::parallel_for(p4_mid, copy_back_mid);
    Kokkos::parallel_for(p4_int, copy_back_int);
  }

  template<typename MyST = ST>
  std::enable_if_t<not std::is_same_v<MyST, DxFadTypeHypervis>>
  run_JtV (const int, const ElementsST<ST>&, ElementsStateST<Real>&) = delete;

  template<typename MyST = ST>
  std::enable_if_t<std::is_same_v<MyST, DxFadTypeHypervis>>
  run_JtV (const int np1, const ElementsST<ST>& e, ElementsStateST<Real>& adj_state)
  {
    const int nelem = e.m_state.num_elems();

    auto dV_v   = ekat::scalarize(e.m_state.m_v);
    auto ddp_v  = ekat::scalarize(e.m_state.m_dp3d);
    auto dvth_v = ekat::scalarize(e.m_state.m_vtheta_dp);
    auto dw_v   = ekat::scalarize(e.m_state.m_w_i);
    auto dphi_v = ekat::scalarize(e.m_state.m_phinh_i);

    auto l_V    = ekat::scalarize(adj_state.m_v);
    auto l_dp   = ekat::scalarize(adj_state.m_dp3d);
    auto l_vth  = ekat::scalarize(adj_state.m_vtheta_dp);
    auto l_w    = ekat::scalarize(adj_state.m_w_i);
    auto l_phi  = ekat::scalarize(adj_state.m_phinh_i);

    ExecViewManaged<Real*[2][NP][NP][NUM_LEV]> out_V("", nelem);
    ExecViewManaged<Real*[NP][NP][NUM_LEV]> out_dp("", nelem);
    ExecViewManaged<Real*[NP][NP][NUM_LEV]> out_vth("", nelem);
    ExecViewManaged<Real*[NP][NP][NUM_LEV_P]> out_w_pack("", nelem);
    ExecViewManaged<Real*[NP][NP][NUM_LEV_P]> out_phi_pack("", nelem);
    auto out_w = ekat::scalarize(out_w_pack);
    auto out_phi = ekat::scalarize(out_phi_pack);

    using md_range_t = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<4>>;
    auto p4_mid = md_range_t({0,0,0,0}, {nelem,NP,NP,NUM_PHYSICAL_LEV});
    auto p4_int = md_range_t({0,0,0,0}, {nelem,NP,NP,NUM_INTERFACE_LEV});

    // JtV (transpose) for midpoint variables.
    // Block-diagonal: only u/v outputs contribute to the u/v adjoint (Group A),
    // and only vth/dp outputs contribute to the vth/dp adjoint (Group B).
    // Slot gp_idx       carries d/d(first-var-in-group)[gp]
    // Slot NP*NP+gp_idx carries d/d(second-var-in-group)[gp]
    auto jtv_mid = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
      const int gp_idx = ip*NP + jp;

      const auto Ju   = Homme::subview(dV_v  ,ie,np1,0);
      const auto Jv   = Homme::subview(dV_v  ,ie,np1,1);
      const auto Jdp  = Homme::subview(ddp_v ,ie,np1);
      const auto Jvth = Homme::subview(dvth_v,ie,np1);

      Real l_u_new   = 0;
      Real l_v_new   = 0;
      Real l_dp_new  = 0;
      Real l_vth_new = 0;
      for (int m = 0; m < NP; ++m) {
        for (int n = 0; n < NP; ++n) {
          // Group A: u-adjoint from slot gp_idx, v-adjoint from slot NP*NP+gp_idx
          // Only u,v outputs carry non-zero derivs at these slots (block-diagonal)
          l_u_new   += Ju  (m,n,k).dx(gp_idx      ) * l_V  (ie,np1,0,m,n,k)
                    +  Jv  (m,n,k).dx(gp_idx      ) * l_V  (ie,np1,1,m,n,k);
          l_v_new   += Ju  (m,n,k).dx(NP*NP+gp_idx) * l_V  (ie,np1,0,m,n,k)
                    +  Jv  (m,n,k).dx(NP*NP+gp_idx) * l_V  (ie,np1,1,m,n,k);
          // Group B: vth-adjoint from slot gp_idx, dp-adjoint from slot NP*NP+gp_idx
          // Only vth,dp outputs carry non-zero derivs at these slots (block-diagonal)
          l_vth_new += Jvth(m,n,k).dx(gp_idx      ) * l_vth(ie,np1,m,n,k)
                    +  Jdp (m,n,k).dx(gp_idx      ) * l_dp (ie,np1,m,n,k);
          l_dp_new  += Jvth(m,n,k).dx(NP*NP+gp_idx) * l_vth(ie,np1,m,n,k)
                    +  Jdp (m,n,k).dx(NP*NP+gp_idx) * l_dp (ie,np1,m,n,k);
        }
      }
      out_V  (ie,0,ip,jp,k) = l_u_new;
      out_V  (ie,1,ip,jp,k) = l_v_new;
      out_dp (ie,ip,jp,k) = l_dp_new;
      out_vth(ie,ip,jp,k) = l_vth_new;
    };

    // JtV (transpose) for interface variables.
    // Group C {w_i}: interior levels — only w output contributes at slot gp_idx.
    //   Surface level — run_w_surf overwrites w_surf with f(u,v), so w_init at the
    //   surface is never read; its adjoint is 0.
    // Group D {phi}: only phi output contributes at slot gp_idx at all levels.
    auto jtv_int = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
      const int gp_idx = ip*NP + jp;
      const bool is_surf = (k >= NUM_PHYSICAL_LEV);

      const auto Jw   = Homme::subview(dw_v  ,ie,np1);
      const auto Jphi = Homme::subview(dphi_v,ie,np1);

      Real l_w_new   = 0;
      Real l_phi_new = 0;
      for (int m = 0; m < NP; ++m) {
        for (int n = 0; n < NP; ++n) {
          if (!is_surf) {
            // Interior w (Group C): gradient w.r.t. w_init[ip,jp] at slot gp_idx
            l_w_new += Jw(m,n,k).dx(gp_idx) * l_w(ie,np1,m,n,k);
          }
          // Surface w: w_init at the surface is overwritten by run_w_surf and never
          // used, so the adjoint l_w_new remains 0 (left unchanged above).
          // phi (Group D): gradient w.r.t. phi_init[ip,jp] at slot gp_idx
          l_phi_new += Jphi(m,n,k).dx(gp_idx) * l_phi(ie,np1,m,n,k);
        }
      }
      out_w  (ie,ip,jp,k) = l_w_new;
      out_phi(ie,ip,jp,k) = l_phi_new;
    };

    auto copy_back_mid = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
      l_V  (ie,np1,0,ip,jp,k) = out_V  (ie,0,ip,jp,k);
      l_V  (ie,np1,1,ip,jp,k) = out_V  (ie,1,ip,jp,k);
      l_dp (ie,np1,ip,jp,k) = out_dp (ie,ip,jp,k);
      l_vth(ie,np1,ip,jp,k) = out_vth(ie,ip,jp,k);
    };

    auto copy_back_int = KOKKOS_LAMBDA (const int ie, const int ip, const int jp, const int k) {
      l_w  (ie,np1,ip,jp,k) = out_w  (ie,ip,jp,k);
      l_phi(ie,np1,ip,jp,k) = out_phi(ie,ip,jp,k);
    };

    Kokkos::parallel_for(p4_mid, jtv_mid);
    Kokkos::parallel_for(p4_int, jtv_int);
    Kokkos::parallel_for(p4_mid, copy_back_mid);
    Kokkos::parallel_for(p4_int, copy_back_int);
  }
#endif // HOMMEXX_ENABLE_FAD_TYPES

  // first iter of laplace, const hv
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagFirstLaplaceHV&, const TeamMember& team) const {
    using Column = decltype(Homme::subview(m_state.m_phinh_i,0,0,0,0));
    using RefColumn = decltype(Homme::subview(m_state.m_ref_states.phi_i_ref,0,0,0));

    KernelVariables kv(team, m_tu);
    // Subtract the reference states from the states
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      auto vtheta = Homme::subview(m_state.m_vtheta_dp,kv.ie,m_data.np1,igp,jgp);
      auto dp    = Homme::subview(m_state.m_dp3d,kv.ie,m_data.np1,igp,jgp);

      auto theta_ref = Homme::subview(m_state.m_ref_states.theta_ref,kv.ie,igp,jgp);
      auto dp_ref    = Homme::subview(m_state.m_ref_states.dp_ref,kv.ie,igp,jgp);

      Column phi_i;
      RefColumn phi_i_ref;

      if (m_process_nh_vars) {
        phi_i = Homme::subview(m_state.m_phinh_i,kv.ie,m_data.np1,igp,jgp);
        phi_i_ref = Homme::subview(m_state.m_ref_states.phi_i_ref,kv.ie,igp,jgp);
      }
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {
        vtheta(ilev) -= theta_ref(ilev);
        dp(ilev)     -= dp_ref(ilev);
        if (m_process_nh_vars) {
          phi_i(ilev)  -= phi_i_ref(ilev);
        }
      });

//defined/used where?
#ifndef XX_NONBFB_COMING
      // It would be fine to not even bother with the surface level, since
      // phitens is only NUM_LEV long, so all the hv stuff does not even happen
      // at NUM_LEV_P (unless NUM_LEV_P==NUM_LEV). However, removing the subtraction
      // and addition of phi_i_ref at NUM_LEV_P introduces NON BFB diffs.
      if (m_process_nh_vars && NUM_LEV!=NUM_LEV_P) {
        Kokkos::single(Kokkos::PerThread(kv.team),[&](){
          phi_i(NUM_LEV_P-1) -= phi_i_ref(NUM_LEV_P-1);
        });
      }
#endif
    }); //team thread range

    //to ensure profiles are fully subtracted
    kv.team_barrier();

    // Laplacian of layer thickness
    m_sphere_ops.laplace_simple(kv,
                   Homme::subview(m_state.m_dp3d,kv.ie,m_data.np1),
                   Homme::subview(m_buffers.dptens,kv.ie));
    // Laplacian of theta
    m_sphere_ops.laplace_simple(kv,
                   Homme::subview(m_state.m_vtheta_dp,kv.ie,m_data.np1),
                   Homme::subview(m_buffers.ttens,kv.ie));

    if (m_process_nh_vars) {
      // Laplacian of vertical velocity (do not compute last interface)
      m_sphere_ops.template laplace_simple<NUM_LEV,NUM_LEV_P>(kv,
                     Homme::subview(m_state.m_w_i,kv.ie,m_data.np1),
                     Homme::subview(m_buffers.wtens,kv.ie));
      // Laplacian of geopotential (do not compute last interface)
      m_sphere_ops.template laplace_simple<NUM_LEV,NUM_LEV_P>(kv,
                     Homme::subview(m_state.m_phinh_i,kv.ie,m_data.np1),
                     Homme::subview(m_buffers.phitens,kv.ie));
    }//if

    // Laplacian of velocity
    m_sphere_ops.vlaplace_sphere_wk_contra(kv, m_data.nu_ratio1,
                              Homme::subview(m_state.m_v,kv.ie,m_data.np1),
                              Homme::subview(m_buffers.vtens,kv.ie));
  }//TagFirstLaplaceHV

  // Laplace for nu_top
  KOKKOS_INLINE_FUNCTION
  void operator()(const TagNutopLaplace&, const TeamMember& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagNutopUpdateStates&, const TeamMember& team) const;

  //second iter of laplace, const hv
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagSecondLaplaceConstHV&, const TeamMember& team) const {
    KernelVariables kv(team, m_tu);
    // Laplacian of layers thickness
    m_sphere_ops.laplace_simple(kv,
                   Homme::subview(m_buffers.dptens,kv.ie),
                   Homme::subview(m_buffers.dptens,kv.ie));
    // Laplacian of theta
    m_sphere_ops.laplace_simple(kv,
                   Homme::subview(m_buffers.ttens,kv.ie),
                   Homme::subview(m_buffers.ttens,kv.ie));

    if (m_process_nh_vars) {
      // Laplacian of vertical velocity
      m_sphere_ops.laplace_simple(kv,
                     Homme::subview(m_buffers.wtens,kv.ie),
                     Homme::subview(m_buffers.wtens,kv.ie));
      // Laplacian of vertical geopotential
      m_sphere_ops.laplace_simple(kv,
                     Homme::subview(m_buffers.phitens,kv.ie),
                     Homme::subview(m_buffers.phitens,kv.ie));
    }
    // Laplacian of velocity
    m_sphere_ops.vlaplace_sphere_wk_contra(kv, m_data.nu_ratio2,
                              Homme::subview(m_buffers.vtens,kv.ie),
                              Homme::subview(m_buffers.vtens,kv.ie));
  } //tag second laplace const hv

  //second iter of laplace, tensor hv
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagSecondLaplaceTensorHV&, const TeamMember& team) const {
    KernelVariables kv(team, m_tu);
    // Laplacian of layers thickness
    m_sphere_ops.laplace_tensor(kv,
                   Homme::subview(m_geometry.m_tensorvisc,kv.ie),
                   Homme::subview(m_buffers.dptens,kv.ie),
                   Homme::subview(m_buffers.dptens,kv.ie));
    // Laplacian of theta
    m_sphere_ops.laplace_tensor(kv,
                   Homme::subview(m_geometry.m_tensorvisc,kv.ie),
                   Homme::subview(m_buffers.ttens,kv.ie),
                   Homme::subview(m_buffers.ttens,kv.ie));

    if (m_process_nh_vars) {
      // Laplacian of vertical velocity
      m_sphere_ops.laplace_tensor(kv,
                     Homme::subview(m_geometry.m_tensorvisc,kv.ie),
                     Homme::subview(m_buffers.wtens,kv.ie),
                     Homme::subview(m_buffers.wtens,kv.ie));
      // Laplacian of geopotential
      m_sphere_ops.laplace_tensor(kv,
                     Homme::subview(m_geometry.m_tensorvisc,kv.ie),
                     Homme::subview(m_buffers.phitens,kv.ie),
                     Homme::subview(m_buffers.phitens,kv.ie));
    }

    m_sphere_ops.vlaplace_sphere_wk_cartesian(kv, 
                   Homme::subview(m_geometry.m_tensorvisc,kv.ie),
                   Homme::subview(m_geometry.m_vec_sph2cart,kv.ie),
                   Homme::subview(m_buffers.vtens,kv.ie),
                   Homme::subview(m_buffers.vtens,kv.ie));
  } //SecondLaplaceTensorHV

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagUpdateStates&, const TeamMember& team) const {
    KernelVariables kv(team, m_tu);

    using MidColumn = decltype(Homme::subview(m_buffers.wtens,0,0,0));
    using IntColumn = decltype(Homme::subview(m_state.m_w_i,0,0,0,0));
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,NP*NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      // Add Xtens quantities back to the states, except for vtheta
      auto u = Homme::subview(m_state.m_v,kv.ie,m_data.np1,0,igp,jgp);
      auto v = Homme::subview(m_state.m_v,kv.ie,m_data.np1,1,igp,jgp);
      auto vtheta = Homme::subview(m_state.m_vtheta_dp,kv.ie,m_data.np1,igp,jgp);
      auto dp     = Homme::subview(m_state.m_dp3d,kv.ie,m_data.np1,igp,jgp);

      auto utens   = Homme::subview(m_buffers.vtens,kv.ie,0,igp,jgp);
      auto vtens   = Homme::subview(m_buffers.vtens,kv.ie,1,igp,jgp);
      auto ttens   = Homme::subview(m_buffers.ttens,kv.ie,igp,jgp);
      auto dptens  = Homme::subview(m_buffers.dptens,kv.ie,igp,jgp);
      const auto& rspheremp = m_geometry.m_rspheremp(kv.ie,igp,jgp);

      MidColumn wtens, phitens;
      IntColumn w, phi_i;

      if (m_process_nh_vars) {
        wtens   = Homme::subview(m_buffers.wtens,kv.ie,igp,jgp);
        phitens = Homme::subview(m_buffers.phitens,kv.ie,igp,jgp);
        w       = Homme::subview(m_state.m_w_i,kv.ie,m_data.np1,igp,jgp);
        phi_i   = Homme::subview(m_state.m_phinh_i,kv.ie,m_data.np1,igp,jgp);
      }

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {

        utens(ilev)   *= m_data.dt_hvs*rspheremp;
        vtens(ilev)   *= m_data.dt_hvs*rspheremp;
        ttens(ilev)   *= m_data.dt_hvs*rspheremp;
        dptens(ilev)  *= m_data.dt_hvs*rspheremp;


        u(ilev)      += utens(ilev);
        v(ilev)      += vtens(ilev);
        vtheta(ilev) += ttens(ilev);
        dp(ilev)     += dptens(ilev);
        if (m_process_nh_vars) {
          wtens(ilev)   *= m_data.dt_hvs * rspheremp;
          phitens(ilev) *= m_data.dt_hvs * rspheremp;

          w(ilev)      += wtens(ilev);
          phi_i(ilev)  += phitens(ilev);
        }
      });
    });
  }  //tagupdatestates

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagHyperPreExchange, const TeamMember &team) const {
    using Column = decltype(Homme::subview(m_state.m_phinh_i,0,0,0,0));
    using RefColumn = decltype(Homme::subview(m_state.m_ref_states.phi_i_ref,0,0,0));

    KernelVariables kv(team, m_tu);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &point_idx) {
      const int igp = point_idx / NP;
      const int jgp = point_idx % NP;

      const auto dpdiss_ave = Homme::subview(m_derived.m_dpdiss_ave,kv.ie, igp, jgp);
      const auto dpdiss_bih = Homme::subview(m_derived.m_dpdiss_biharmonic,kv.ie, igp, jgp);
      const auto dp3d = Homme::subview(m_state.m_dp3d,kv.ie, m_data.np1, igp, jgp);
      const auto dp_ref = Homme::subview(m_state.m_ref_states.dp_ref,kv.ie, igp, jgp);
      const auto theta = Homme::subview(m_state.m_vtheta_dp,kv.ie, m_data.np1, igp, jgp);
      const auto theta_ref = Homme::subview(m_state.m_ref_states.theta_ref,kv.ie, igp, jgp);
      const auto dptens = Homme::subview(m_buffers.dptens,kv.ie, igp, jgp);

      Column phi;
      RefColumn phi_ref;

      if (m_process_nh_vars) {
        phi = Homme::subview(m_state.m_phinh_i,kv.ie, m_data.np1, igp, jgp);
        phi_ref = Homme::subview(m_state.m_ref_states.phi_i_ref,kv.ie, igp, jgp);
      }

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                           [&](const int &ilev) {

        dp3d(ilev) += dp_ref(ilev);
        theta(ilev) += theta_ref(ilev);
        if (m_process_nh_vars) {
          phi(ilev) += phi_ref(ilev);
        }
        if (m_data.nu_p>0) {
          dpdiss_ave(ilev) += m_data.eta_ave_w*dp3d(ilev) / m_data.hypervis_subcycle;
          dpdiss_bih(ilev) += m_data.eta_ave_w*dptens(ilev) / m_data.hypervis_subcycle;
        }
      });

//where is it set?
#ifndef XX_NONBFB_COMING
      // It would be fine to not even bother with the surface level, since
      // phitens is only NUM_LEV long, so all the hv stuff does not even happen
      // at NUM_LEV_P (unless NUM_LEV_P==NUM_LEV). However, removing the subtraction
      // and addition of phi_i_ref at NUM_LEV_P introduces NON BFB diffs.
      if (m_process_nh_vars && NUM_LEV!=NUM_LEV_P) {
        Kokkos::single(Kokkos::PerThread(kv.team),[&](){
          phi(NUM_LEV_P-1) += phi_ref(NUM_LEV_P-1);
        });
      }
#endif
    });//teamthreadrange loop
    kv.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &point_idx) {
      const int igp = point_idx / NP;
      const int jgp = point_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                           [&](const int &lev) {
        m_buffers.vtens(kv.ie, 0, igp, jgp, lev) *= -m_data.nu;
        m_buffers.vtens(kv.ie, 1, igp, jgp, lev) *= -m_data.nu;
        m_buffers.ttens(kv.ie, igp, jgp, lev) *= -m_data.nu;
        m_buffers.dptens(kv.ie, igp, jgp, lev) *= -m_data.nu_p;
        if (m_process_nh_vars) {
          m_buffers.wtens(kv.ie, igp, jgp, lev) *= -m_data.nu;
          m_buffers.phitens(kv.ie, igp, jgp, lev) *= -m_data.nu_s;
        }
      });//thread vector

    });//parallel 4
  } //taghyperpreexchange

protected:

  const int                   m_num_elems;
  HyperviscosityData          m_data;
  ElementsStateST<ST>         m_state;
  ElementsDerivedStateST<ST>  m_derived;
  ElementsGeometry            m_geometry;
  SphereOperatorsST<ST>       m_sphere_ops;
  ElementOps                  m_elem_ops;
  EquationOfState<>           m_eos;
  Buffers                     m_buffers;
  HybridVCoord                m_hvcoord;

  bool m_process_nh_vars;

  // Policies
  Kokkos::TeamPolicy<ExecSpace,TagUpdateStates>     m_policy_update_states;
  Kokkos::TeamPolicy<ExecSpace,TagFirstLaplaceHV>   m_policy_first_laplace;
  Kokkos::TeamPolicy<ExecSpace,TagHyperPreExchange> m_policy_pre_exchange;

  Kokkos::TeamPolicy<ExecSpace,TagNutopLaplace>      m_policy_nutop_laplace;
  Kokkos::TeamPolicy<ExecSpace,TagNutopUpdateStates> m_policy_nutop_update_states;

  TeamUtils<ExecSpace> m_tu; // If the policies only differ by tag, just need one tu

  std::shared_ptr<BoundaryExchangeST<ST>> m_be, m_be_tom;

  ExecViewManaged<PT[NUM_LEV]> m_nu_scale_top;
  int m_nu_scale_top_ilev_pack_lim;
}; //HVfunctorImplST

using HyperviscosityFunctorImpl = HyperviscosityFunctorImplST<ScalarValue>;

} // namespace Homme

#endif // HOMMEXX_HYPERVISCOSITY_FUNCTOR_IMPL_HPP
