/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "CaarFunctor.hpp"
#include "LimiterFunctor.hpp"
#include "DirkFunctor.hpp"
#include "Context.hpp"
#include "Diagnostics.hpp"
#include "Elements.hpp"
#include "HyperviscosityFunctor.hpp"
#include "PhysicalConstants.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "mpi/BoundaryExchange.hpp"

#include "profiling.hpp"

namespace Homme
{

struct AdjBE : public std::vector<BoundaryExchangeST<Real>> {};

void init_adj_state_be (ElementsStateST<Real>& adj_state)
{
  auto& c = Context::singleton();
  if (c.has<AdjBE>())
    return;

  const auto& params = c.get<SimulationParams>();
  auto& bmm = c.get<MpiBuffersManagerMap>();

  auto& be = c.create<AdjBE>(NUM_TIME_LEVELS);
  for (int itl=0; itl<NUM_TIME_LEVELS; ++itl) {
    be[itl].m_label = std::string("AdjState-") + std::to_string(itl);
    be[itl].set_buffers_manager(bmm[MPI_EXCHANGE]);
    be[itl].m_diagnostics_level = params.internal_diagnostics_level;
    if (params.theta_hydrostatic_mode) {
      be[itl].set_num_fields(0,0,4);
    } else {
      be[itl].set_num_fields(0,0,4,2);
    }
      be[itl].register_field(m_state.m_v,tl,2,0);
      be[itl].register_field(m_state.m_vtheta_dp,1,tl);
      be[itl].register_field(m_state.m_dp3d,1,tl);
      if (!m_theta_hydrostatic_mode) {
        // Note: phinh_i at the surface (last level) is constant, so it doesn't *need* bex.
        //       If bex(constant)=constant, we might just do it. This would not eliminate
        //       the need for halo-exchange of interface-based quantities though, since
        //       we would still need to exchange w_i.
        be[itl].register_field(m_state.m_w_i,1,tl);
        be[itl].register_field(m_state.m_phinh_i,1,tl);
      }
      be[itl].registration_completed();
  }
}

void ttype9_imex_adjoint (const TimeLevel& tl,
                          const Real dt_dyn,
                          const Real eta_ave_w,
                          ElementsStateST<Real>& adj_state)
{
  GPTLstart("ttype9_imex_adjoint");

  const auto& c = Context::singleton();
  SimulationParams& params = c.get<SimulationParams>();

  // Get elements, hvcoord, and functors
  auto& elems_dirk = c.get<ElementsST<DxFadTypeDirk>>();
  auto& elems_caar = c.get<ElementsST<DxFadTypeCaar>>();
  auto& hvcoord  = c.get<HybridVCoord>();
  auto& dirk     = c.get<DirkFunctorST<DxFadTypeDirk>>();
  auto& caar     = c.get<CaarFunctorST<DxFadTypeDirk>>();
  // auto& limiter  = c.get<LimiterFunctor>();
  auto& tape     = c.get<ImexTape>();

  int nelem = adj_state.num_elems();
  int nm1 = 0;
  int n0  = 1;
  int np1 = 2;

  // For each functor, load fwd state we had right before
  // running it, run functor, then compute JtV (with V=adj_state)
  StateSnapshot snap_n0(nelem), snap_nm1(nelem);

  // NOTATION:
  //
  // State:
  //  - u_i: state after explicit CAAR stage
  //  - y_i: state after implicit DIRK stage
  // where y_0 is the state at the beginning of prim_advance_exp,
  // and y_5 is the state at the end (after 5th DIRK stage)
  //
  // Adjoint state:
  //  - lambda_i: deriv w.r.t. u_i
  //  - mu_i: deriv w.r.t. y_i



//   ////////////////////////////////////////////////////////////
//   snap_n0  = tape.pop_back();
//   snap_nm1 = tape.pop_back();

//   state.import_snapshot(snap_n0, n0);
//   state.import_snapshot(snap_nm1,nm1);

//   Real a1 = 5.0*dt_dyn/18.0;
//   Real a2 = dt_dyn/36.0;
//   Real a3 = 8.0*dt_dyn/18.0;

//   dirk.init_J(n0,elems_dirk);
//   dirk.run(nm1, a2, n0, a1, np1, a3, elems_dirk, hvcoord);
//   dirk.run_JtV(n0,np1,elems_dirk,adj_state);

//   ////////////////////////////////////////////////////////////
//   const int nm1 = tl.nm1;
//   const int n0  = tl.n0;
//   const int np1 = tl.np1;
//   const int qn0 = tl.n0_qdp;

//   auto save = [&](int tl) {
//     if (not params.store_fwd_state)
//       return;

//     auto& tape = c.get<StateTape>();
//     auto& snap = tape.emplace_back();
//     elements.m_state.take_snapshot(tl);
//   };

//   // Stage 1
//   Real dt = dt_dyn/5.0;

// // subroutine compute_andor_apply_rhs(np1,nm1,n0,dt2,...
// //
// // if one wants to map F call compute_andor_apply_rhs(n1,n2,n3,qn0,...
// // to caar call below, then they need to use timelevels this way
// //         caar(n2,n3,n1,qn0,...
// //
// // Names of timelevels in RK:
// //         RKStageData (const int nm1_in, const int n0_in, const int np1_in, const int n0_qdp_in ...
//   caar.run(RKStageData(n0, n0, nm1, qn0, dt, eta_ave_w/4.0, 1.0, 0.0, 1.0));
//   save(nm1);
//   dirk.run(nm1, 0.0, n0, 0.0, nm1, dt, elements, hvcoord);
//   save(nm1);

//   // Stage 2
//   dt = dt_dyn/5.0;
//   caar.run(RKStageData(n0, nm1, np1, qn0, dt, 0.0, 1.0, 0.0, 1.0));
//   save(np1);
//   dirk.run(nm1, 0.0, n0, 0.0, np1, dt, elements, hvcoord);
//   save(np1);

//   // Stage 3
//   dt = dt_dyn/3.0;
//   caar.run(RKStageData(n0, np1, np1, qn0, dt, 0.0, 1.0, 0.0, 1.0));
//   save(np1);
//   dirk.run(nm1, 0.0, n0, 0.0, np1, dt, elements, hvcoord);
//   save(np1);

//   // Stage 4
//   dt = 2.0*dt_dyn/3.0;
//   caar.run(RKStageData(n0, np1, np1, qn0, dt, 0.0, 1.0, 0.0, 1.0));
//   save(np1);
//   dirk.run(nm1, 0.0, n0, 0.0, np1, dt, elements, hvcoord);
//   save(np1);

//   // Stage 5
//   dt = 3.0*dt_dyn/4.0;
//   caar.run(RKStageData(nm1, np1, np1, qn0, dt, 3.0*eta_ave_w/4.0, 1.0, 0.0, 1.0));
//   save(np1);
//   // u(np1) = [u1 + 3dt/4 RHS(u4)] +  1/4 (u1 - u0)
//   { 
//     const auto v         = elements.m_state.m_v;
//     const auto w         = elements.m_state.m_w_i;
//     const auto vtheta_dp = elements.m_state.m_vtheta_dp;
//     const auto phinh     = elements.m_state.m_phinh_i;
//     const auto dp3d      = elements.m_state.m_dp3d;
//     const auto hydrostatic_mode = params.theta_hydrostatic_mode;
    
//     Kokkos::parallel_for(
//       Kokkos::RangePolicy<ExecSpace>(0, elements.num_elems()*NP*NP*NUM_LEV),
//       KOKKOS_LAMBDA(const int it) {
//         const int ie = it / (NP*NP*NUM_LEV);
//         const int igp = (it / (NP*NUM_LEV)) % NP;
//         const int jgp = (it / NUM_LEV) % NP;
//         const int ilev = it % NUM_LEV;
//         v(ie,np1,0,igp,jgp,ilev) += (v(ie,nm1,0,igp,jgp,ilev)-v(ie,n0,0,igp,jgp,ilev))/4.0;
//         v(ie,np1,1,igp,jgp,ilev) += (v(ie,nm1,1,igp,jgp,ilev)-v(ie,n0,1,igp,jgp,ilev))/4.0;
//         vtheta_dp(ie,np1,igp,jgp,ilev) += (vtheta_dp(ie,nm1,igp,jgp,ilev)-vtheta_dp(ie,n0,igp,jgp,ilev))/4.0;
//         dp3d(ie,np1,igp,jgp,ilev)      += (dp3d(ie,nm1,igp,jgp,ilev)-dp3d(ie,n0,igp,jgp,ilev))/4.0;
//         if (!hydrostatic_mode) { 
//           w(ie,np1,igp,jgp,ilev)       += (w(ie,nm1,igp,jgp,ilev)-w(ie,n0,igp,jgp,ilev))/4.0;
//           phinh(ie,np1,igp,jgp,ilev)   += (phinh(ie,nm1,igp,jgp,ilev)-phinh(ie,n0,igp,jgp,ilev))/4.0;
//         }
//     });
//     if (NUM_LEV_P>NUM_LEV && !hydrostatic_mode) {
//       const int LAST_INT = NUM_LEV_P-1;
//       Kokkos::parallel_for(
//         Kokkos::RangePolicy<ExecSpace>(0, elements.num_elems()*NP*NP),
//         KOKKOS_LAMBDA(const int it) {
//            const int ie  =  it / (NP*NP);
//            const int igp = (it / NP) % NP;
//            const int jgp =  it % NP;
//            w(ie,np1,igp,jgp,LAST_INT)  += (w(ie,nm1,igp,jgp,LAST_INT)-w(ie,n0,igp,jgp,LAST_INT))/4.0;
//       });  
//     }
//   }
//   Kokkos::fence();
//   save(np1);
//   limiter.run(np1);
//   save(np1);

//   Real a1 = 5.0*dt_dyn/18.0;
//   Real a2 = dt_dyn/36.0;
//   Real a3 = 8.0*dt_dyn/18.0;
//   dirk.run(nm1, a2, n0, a1, np1, a3, elements, hvcoord);
//   save(np1);

  GPTLstop("ttype9_imex_adjoint");
}

} // namespace Homme
