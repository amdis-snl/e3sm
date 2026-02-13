#ifndef HOMMEXX_ELEMENTS_FORCING_HPP
#define HOMMEXX_ELEMENTS_FORCING_HPP

#include <Types.hpp>

namespace Homme {

template<typename ST>
class ElementsForcingST {
public:
  using PT = PackType<ST>;

  // Per Element Forcings
  ExecViewManaged<PT * [3][NP][NP][NUM_LEV]  >  m_fm;       // Momentum forcing
  ExecViewManaged<PT *    [NP][NP][NUM_LEV]  >  m_fvtheta;  // Virtual potential temperature forcing
  ExecViewManaged<PT *    [NP][NP][NUM_LEV]  >  m_ft;       // Temperature forcing
  ExecViewManaged<PT *    [NP][NP][NUM_LEV_P]>  m_fphi;     // Phi (NH) forcing

  ElementsForcingST() = default;

  void init (const int num_elems);
  void randomize (const int seed, const Real min_f = -1.0, const Real max_f = 1.0);

  int num_elems () const { return m_num_elems; }

private:
  int m_num_elems;
};

using ElementsForcing = ElementsForcingST<ScalarValue>;

} // namespace Homme

#endif // HOMMEXX_ELEMENTS_FORCING_HPP
