#include "ElementsForcing.hpp"
#include "ElementsForcing_def.hpp"

#include "Types.hpp"

namespace Homme {

// ETI for the commonly used scalar type
template class ElementsForcingST<Real>;

#ifdef HOMMEXX_ENABLE_FAD_TYPES
template class ElementsForcingST<DpFadType>;
template class ElementsForcingST<DxFadTypeCaar>;
#endif

} // namespace Homme
