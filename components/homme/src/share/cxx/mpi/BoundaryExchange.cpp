/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "BoundaryExchange.hpp"
#include "BoundaryExchange_def.hpp"

#include "Types.hpp"

namespace Homme {

// ETI for the commonly used scalar type
template class BoundaryExchangeST<Real>;

#ifdef HOMMEXX_ENABLE_FAD_TYPES
template class BoundaryExchangeST<DpFadType>;
template class BoundaryExchangeST<DxFadTypeCaar>;
template class BoundaryExchangeST<DxFadTypeHypervis>;
#endif

} // namespace Homme
