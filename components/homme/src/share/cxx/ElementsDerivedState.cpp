/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ElementsDerivedState.hpp"
#include "ElementsDerivedState_def.hpp"

#include "Types.hpp"

namespace Homme {

// ETI for the commonly used scalar type
template class ElementsDerivedStateST<Real>;

#ifdef HOMMEXX_ENABLE_FAD_TYPES
template class ElementsDerivedStateST<DpFadType>;
template class ElementsDerivedStateST<DxFadTypeCaar>;
#endif

} // namespace Homme
