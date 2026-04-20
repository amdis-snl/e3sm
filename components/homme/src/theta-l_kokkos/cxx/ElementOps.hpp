#ifndef HOMMEXX_ELEMENT_OPS_HPP
#define HOMMEXX_ELEMENT_OPS_HPP

#include "Types.hpp"
#include "KernelVariables.hpp"
#include "HybridVCoord.hpp"
#include "ColumnOps.hpp"
#include "PhysicalConstants.hpp"

#include "utilities/BfbUtils.hpp"

namespace Homme {

class ElementOps {
public:
  ElementOps () = default;

  ~ElementOps () = default;

  void init (const HybridVCoord& hvcoord) {
    m_hvcoord = hvcoord;
  }

  template<typename InputProvider, typename PT, typename... Props>
  KOKKOS_INLINE_FUNCTION
  void get_R_star (const KernelVariables& kv,
                   const bool use_moisture,
                   const InputProvider& Q,
                   const ExecView<PT[NUM_LEV],Props...>& R) const {
    using namespace PhysicalConstants;
    if (use_moisture) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {
        R(ilev) = (Rgas + (Rwater_vapor-Rgas)*Q(ilev));
      });
    } else {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team,NUM_LEV),
                           [&](const int ilev) {
        R(ilev) = Rgas;
      });
    }
  }

  template<typename InputProvider, typename PT, typename... Props>
  KOKKOS_INLINE_FUNCTION
  void compute_hydrostatic_p (const KernelVariables& kv,
                              const InputProvider& dp,
                              const ExecView<PT[NUM_LEV_P],Props...>& p_i,
                              const ExecView<PT[NUM_LEV  ],Props...>& pi) const
  {
    p_i(0)[0] = m_hvcoord.hybrid_ai0*m_hvcoord.ps0;
    ColumnOps::column_scan_mid_to_int<true>(kv,dp,p_i);
    ColumnOps::compute_midpoint_values(kv,p_i,pi);
  }

  template<typename InputProvider, typename PT, typename... Props>
  KOKKOS_FUNCTION
  void get_temperature (const KernelVariables& kv,
                        const bool use_moisture,
                        const InputProvider& dp,
                        const InputProvider& exner,
                        const InputProvider& vtheta_dp,
                        const InputProvider& qv,
                        const ExecView<PT[NUM_LEV],Props...>& Rstar,
                        const ExecView<PT[NUM_LEV],Props...>& T) const {
    using namespace PhysicalConstants;
    get_R_star(kv, use_moisture, qv, Rstar);
    Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
      [&] (const int k) { T(k) = Rgas * vtheta_dp(k) * exner(k) / (Rstar(k) * dp(k)); });
  }

private:

  HybridVCoord    m_hvcoord;
};

} // namespace Homme

#endif // HOMMEXX_ELEMENT_OPS_HPP
