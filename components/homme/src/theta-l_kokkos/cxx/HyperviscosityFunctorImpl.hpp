/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_HYPERVISCOSITY_FUNCTOR_IMPL_HPP
#define HOMMEXX_HYPERVISCOSITY_FUNCTOR_IMPL_HPP

#include "ElementsGeometry.hpp"
#include "ElementsState.hpp"
#include "ElementsDerivedState.hpp"
#include "ColumnOps.hpp"
#include "EquationOfState.hpp"
#include "ElementOps.hpp"
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
  template<typename Tag>
  using MDRange = Kokkos::MDRangePolicy<Tag,ExecSpace,Kokkos::Rank<4>>;

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

  struct TagPreprocess {};
  struct TagPostprocess {};
  struct TagFirstLaplace {};
  struct TagSecondLaplaceConstHV {};
  struct TagSecondLaplaceTensorHV {};
  struct TagUpdateStates {};

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

  // first iter of laplace, const hv
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagPreprocess&, const int ie, const int igp, const int jgp, const int ilev) const
  {
    auto& vtheta = m_state.m_vtheta_dp(ie,m_data.np1,igp,jgp,ilev);
    auto& dp     = m_state.m_dp3d (ie,m_data.np1,igp,jgp,ilev);

    vtheta /= dp;
    vtheta -= m_state.m_ref_states.theta_ref(ie,igp,jgp,ilev);
    dp -= m_state.m_ref_states.dp_ref(ie,igp,jgp,ilev);

    m_buffers.dptens(ie,igp,jgp,ilev)  = dp;
    m_buffers.ttens(ie,igp,jgp,ilev)   = vtheta;
    m_buffers.vtens(ie,0,igp,jgp,ilev) = m_state.m_v(ie,m_data.np1,0,igp,jgp,ilev);
    m_buffers.vtens(ie,1,igp,jgp,ilev) = m_state.m_v(ie,m_data.np1,1,igp,jgp,ilev);

    if (m_process_nh_vars) {
      m_buffers.phitens = m_state.m_phinh_i(ie,m_data.np1,igp,jgp,ilev) - m_state.m_ref_states.phi_i_ref(ie,igp,jgp,ilev);
      m_buffers.wtens = m_state.m_w_i(ie,m_data.np1,igp,jgp,ilev);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagPostprocess&, const int ie, const int igp, const int jgp, const int ilev) const
  {
    auto& vtheta = m_state.m_vtheta_dp(ie,m_data.np1,igp,jgp,ilev);
    auto& dp     = m_state.m_dp3d (ie,m_data.np1,igp,jgp,ilev);

    vtheta += m_state.m_ref_states.theta_ref(ie,igp,jgp,ilev);
    dp     += m_state.m_ref_states.dp_ref   (ie,igp,jgp,ilev);
    vtheta *= dp;

    dp = m_buffers.dptens(ie,igp,jgp,ilev);
    vtheta = m_buffers.ttens(ie,igp,jgp,ilev);
    m_state.m_v(ie,m_data.np1,0,igp,jgp,ilev) += m_data.dt_hvs*m_buffers.vtens(ie,0,igp,jgp,ilev);
    m_state.m_v(ie,m_data.np1,1,igp,jgp,ilev) += m_data.dt_hvs*m_buffers.vtens(ie,1,igp,jgp,ilev);

    if (m_process_nh_vars) {
      m_state.m_phinh_i (ie,m_data.np1,igp,jgp,ilev) = m_buffers.phitens(ie,igp,jgp,ilev) +  m_state.m_ref_states.phi_i_ref(ie,igp,jgp,ilev);
      m_state.m_w_i (ie,m_data.np1,igp,jgp,ilev) = m_buffers.wtens(ie,igp,jgp,ilev);
      if (ilev==NUM_LEV) {
        using InfoI = ColInfo<NUM_INTERFACE_LEV>;
        using InfoM = ColInfo<NUM_PHYSICAL_LEV>;
        constexpr Real g = PhysicalConstants::g;
        constexpr int LAST_MID_PACK_END = InfoM::LastPackEnd;
        constexpr int LAST_INT_PACK_END = InfoI::LastPackEnd;

        const auto& u = m_state.m_v(ie,m_data.np1,0,igp,jgp,ilev)[LAST_MID_PACK_END];
        const auto& v = m_state.m_v(ie,m_data.np1,1,igp,jgp,ilev)[LAST_MID_PACK_END];
        const auto& grad_x = m_geometry.m_gradphis(ie,0,igp,jgp);
        const auto& grad_y = m_geometry.m_gradphis(ie,1,igp,jgp);
        auto& w = m_state.m_w_i(ie,m_data.np1,igp,jgp,ilev)[LAST_INT_PACK_END];
        w = (u*grad_x+v*grad_y) / g;
      }
    }
  }

  //second iter of laplace, tensor hv
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagLaplaceTensorHV&, const TeamMember& team) const {
    KernelVariables kv(team);
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
  } // LaplaceTensorHV

  // Laplace for nu_top
  KOKKOS_INLINE_FUNCTION
  void operator()(const TagNutopLaplace&, const TeamMember& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagNutopUpdateStates&, const TeamMember& team) const;

  //second iter of laplace, const hv
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagLaplaceConstHV&, const TeamMember& team) const {
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
  MDRange<TagPreprocess>   m_policy_pre_process;
  MDRange<TagPostprocess>  m_policy_post_process;

  Kokkos::TeamPolicy<ExecSpace,TagFirstLaplace>           m_policy_first_laplace;
  Kokkos::TeamPolicy<ExecSpace,TagSecondLaplaceConstHV>   m_policy_second_laplace_const;
  Kokkos::TeamPolicy<ExecSpace,TagSecondLaplaceTensorHV>  m_policy_second_laplace_tensor;

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
