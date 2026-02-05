#ifndef HOMMEXX_ELEMENTS_FORCING_HPP
#define HOMMEXX_ELEMENTS_FORCING_HPP

#include <Types.hpp>

namespace Homme {

template<typename ST>
class ElementsForcingST {
public:
  using PT = PackType<ST>;

  // Per Element Forcings
  ExecViewManaged<PT * [2][NP][NP][NUM_LEV]> m_fm;  // Momentum (? units are wrong in apply_cam_forcing...) forcing
  ExecViewManaged<PT * [NP][NP][NUM_LEV]>    m_ft;  // Temperature forcing

  ElementsForcingST() = default;

  void init(const int num_elems);

private:
  int m_num_elems;
};

using ElementsForcing = ElementsForcingST<ScalarValue>;

} // namespace Homme

#endif // HOMMEXX_ELEMENTS_FORCING_HPP
