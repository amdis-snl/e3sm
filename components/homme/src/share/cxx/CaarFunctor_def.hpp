/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMME_CAAR_FUNCTOR_DEF_HPP
#define HOMME_CAAR_FUNCTOR_DEF_HPP

#include "CaarFunctor.hpp"
#include "CaarFunctorImpl.hpp"
#include "Context.hpp"
#include "ErrorDefs.hpp"
#include "HybridVCoord.hpp"
#include "SimulationParams.hpp"
#include "ReferenceElement.hpp"
#include "Tracers.hpp"
#include "mpi/MpiBuffersManager.hpp"

#include "profiling.hpp"

#include <assert.h>
#include <type_traits>

namespace Homme {

template<typename ST>
CaarFunctorST<ST>::
CaarFunctorST()
  : is_setup(true)
{
  auto& elements   = Context::singleton().get<ElementsST<ST>>();
  auto& tracers    = Context::singleton().get<TracersST<ST>>();
  auto& ref_FE     = Context::singleton().get<ReferenceElement>();
  auto& hvcoord    = Context::singleton().get<HybridVCoord>();
  auto& sphere_ops = Context::singleton().get<SphereOperatorsST<ST>>();
  auto& params     = Context::singleton().get<SimulationParams>();

  // Build functor impl
  m_caar_impl.reset(new CaarFunctorImplST<ST>(elements,tracers,ref_FE,hvcoord,sphere_ops,params));
}

template<typename ST>
CaarFunctorST<ST>::
CaarFunctorST(const ElementsST<ST> &elements, const TracersST<ST> &tracers,
              const ReferenceElement &ref_FE,
              const HybridVCoord &hvcoord,
              const SphereOperatorsST<ST> &sphere_ops,
              const SimulationParams& params)
  : is_setup(true)
{
  // Build functor impl
  m_caar_impl.reset(new CaarFunctorImplST<ST>(elements, tracers, ref_FE, hvcoord, sphere_ops, params));
}

// This constructor is useful for using buffer functionality without
// having all other Functor information available.
// If this constructor is used, the setup() function must be called
// before using any other CaarFunctorST functions.
template<typename ST>
CaarFunctorST<ST>::
CaarFunctorST(const int num_elems, const SimulationParams& params)
  : is_setup(false)
{
  // Build functor impl
  m_caar_impl.reset(new CaarFunctorImplST<ST>(num_elems, params));
}

template<typename ST>
void CaarFunctorST<ST>::
setup(const ElementsST<ST> &elements, const TracersST<ST> &tracers,
      const ReferenceElement &ref_FE, const HybridVCoord &hvcoord,
      const SphereOperatorsST<ST> &sphere_ops)
{
  assert (m_caar_impl);

  // Sanity check
  assert (!is_setup);

  m_caar_impl->setup(elements, tracers, ref_FE, hvcoord, sphere_ops);
  is_setup = true;
}

template<typename ST>
int CaarFunctorST<ST>::requested_buffer_size () const {
  assert (m_caar_impl);
  return m_caar_impl->requested_buffer_size();
}

template<typename ST>
void CaarFunctorST<ST>::init_buffers(const FunctorsBuffersManager& fbm) {
  assert (m_caar_impl);
  m_caar_impl->init_buffers(fbm);
}

template<typename ST>
void CaarFunctorST<ST>::init_boundary_exchanges (const std::shared_ptr<MpiBuffersManager>& bm_exchange) {
  assert (m_caar_impl);

  // The Functor needs to be fully setup to use this function
  assert (is_setup);

  m_caar_impl->init_boundary_exchanges(bm_exchange);
}

template<typename ST>
void CaarFunctorST<ST>::set_rk_stage_data (const RKStageData& data)
{
  // Sanity check (should NEVER happen)
  assert (m_caar_impl);

  // The Functor needs to be fully setup
  assert (is_setup);

  // Forward inputs to impl
  m_caar_impl->set_rk_stage_data(data);
}

template<typename ST>
void CaarFunctorST<ST>::run (const RKStageData& data)
{
  // Sanity check (should NEVER happen)
  assert (m_caar_impl);

  // The Functor needs to be fully setup
  assert (is_setup);

  // Run functor
  m_caar_impl->run(data);
}

} // Namespace Homme

#endif // HOMME_CAAR_FUNCTOR_DEF_HPP
