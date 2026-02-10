/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "EulerStepFunctor.hpp"
#include "EulerStepFunctor_def.hpp"

#include "Types.hpp"

namespace Homme {

// ETI for the commonly used scalar type
template class EulerStepFunctorST<Real>;

#ifdef HOMMEXX_ENABLE_FAD_TYPES
template class EulerStepFunctorST<DpFadType>;
#endif

} // namespace Homme
