/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_TRACERS_HPP
#define HOMMEXX_TRACERS_HPP

#include "Types.hpp"
#include "utilities/Hash.hpp"

namespace Homme {

template<typename ST>
struct TracersST
{
  using PT = PackType<ST>;

  TracersST() : m_inited(false) {}
  TracersST(const int num_elems, const int num_tracers);

  void init (const int num_elems, const int num_tracers);

  void randomize (const int seed, const Real min = -1.0, const Real max = 1.0);

  void pull_qdp(CF90Ptr &state_qdp);
  void push_qdp(F90Ptr &state_qdp) const;

  KOKKOS_INLINE_FUNCTION
  int num_tracers() const {
    return nt;
  }

  int num_elems () const {
    return ne;
  }

  bool inited () const { return m_inited; }

  ExecViewManaged<PT*[Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV]> qdp;
  ExecViewManaged<PT**[NP][NP][NUM_LEV]>                    qtens_biharmonic; // Also doubles as just qtens.
  ExecViewManaged<PT*[QSIZE_D][2][NUM_LEV]>                 qlim;
  ExecViewManaged<PT**[NP][NP][NUM_LEV]>                    Q;
  ExecViewManaged<PT**[NP][NP][NUM_LEV]>                    fq;

  HashType hash(const int qdp_time_level) const;

private:
  int nt;
  int ne;
  bool m_inited;
};

using Tracers = TracersST<ScalarValue>;

} // namespace Homme

#endif // HOMMEXX_TRACERS_HPP
