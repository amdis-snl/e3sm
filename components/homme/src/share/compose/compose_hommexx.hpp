#ifndef INCLUDE_COMPOSE_HOMMEXX_HPP
#define INCLUDE_COMPOSE_HOMMEXX_HPP

#include "compose_slmm_islmpi.hpp"
#include <Kokkos_Core.hpp>

namespace homme {

islmpi::IslMpi<>::Ptr get_isl_mpi_singleton();

namespace compose {

typedef double HommexxReal;

template <typename DataType>
using SetView = Kokkos::View<DataType, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;

typedef SetView<HommexxReal*****> SetView5;

// ETP:  currently a no-op to get the code to link with FadTypes
//       but will need to be handled for semi-lagrangian transport eventually
template <typename Sc>
std::enable_if_t<not std::is_same_v<Sc,HommexxReal>>
set_views(const SetView<HommexxReal***>& spheremp,
          const SetView<Sc****>& dp, const SetView<Sc*****>& dp3d,
          const SetView<Sc******>& qdp, const SetView<Sc*****>& q,
          const SetView<Sc*****>& dep_points, const SetView<Sc*****>& vnode,
          const int ndim, const SetView<Sc*****>& vdep, const int vdep_ndim)
{
  throw std::runtime_error("Error! Should NOT have been called...\n");
}

void set_views(const SetView<HommexxReal***>& spheremp,
               const SetView<HommexxReal****>& dp, const SetView5& dp3d,
               const SetView<HommexxReal******>& qdp, const SetView5& q,
               const SetView5& dep_points, const SetView5& vnode, const int ndim,
               const SetView5& vdep, const int vdep_ndim);

template<typename ST>
void set_hvcoord(const ST* etai, const ST* etam)
{
  auto& cm = *get_isl_mpi_singleton();
  islmpi::set_hvcoord(cm, etai, etam);
}

void interp_v_update(const int step, const HommexxReal dtsub);

void advect(const int np1, const int n0_qdp, const int np1_qdp);

void set_dp3d_np1(const int np1);
bool property_preserve_global();
bool property_preserve_local(const int limiter_option);
void property_preserve_check();

void finalize();

} // namespace compose
} // namespace homme

#endif
