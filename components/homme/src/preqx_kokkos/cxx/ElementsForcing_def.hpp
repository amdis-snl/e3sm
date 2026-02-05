#ifndef HOMME_ELEMENTS_FORCING_DEF_HPP
#define HOMME_ELEMENTS_FORCING_DEF_HPP

#include "ElementsForcing.hpp"

namespace Homme {

template<typename ST>
void ElementsForcing<ST>::init (const int num_elems) {
  m_num_elems = num_elems;

  m_fm = ExecViewManaged<PT * [2][NP][NP][NUM_LEV]>("F_Momentum",    m_num_elems);
  m_ft = ExecViewManaged<PT *    [NP][NP][NUM_LEV]>("F_Temperature", m_num_elems);
}

} // namespace Homme

#endif // HOMME_ELEMENTS_FORCING_DEF_HPP
