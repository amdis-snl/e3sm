/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_EULER_STEP_FUNCTOR_DEF_HPP
#define HOMMEXX_EULER_STEP_FUNCTOR_DEF_HPP

#include "EulerStepFunctorImpl.hpp"

namespace Homme {

template<typename ST>
EulerStepFunctorST<ST>::
EulerStepFunctorST()
  : is_setup(true) {
  p_ = std::make_shared<EulerStepFunctorImplST<ST>>();
}

// This constructor is useful for using buffer functionality without
// having all other Functor information available.
// If this constructor is used, the setup() function must be called
// before using any other EulerStepFunctor functions.
template<typename ST>
EulerStepFunctorST<ST>::
EulerStepFunctorST(const int num_elems)
  : is_setup(false)
{
  p_ = std::make_shared<EulerStepFunctorImplST<ST>>(num_elems);
}

template<typename ST>
void EulerStepFunctorST<ST>::setup()
{
  // Sanity check
  assert (!is_setup);

  p_->setup();
  is_setup = true;
}

template<typename ST>
void EulerStepFunctorST<ST>::reset (const SimulationParams& params) {
  p_->reset(params);
}

template<typename ST>
int EulerStepFunctorST<ST>::requested_buffer_size () const {
  return p_->requested_buffer_size();
}
template<typename ST>
void EulerStepFunctorST<ST>::init_buffers (const FunctorsBuffersManager& fbm) {
  p_->init_buffers(fbm);
}

template<typename ST>
void EulerStepFunctorST<ST>::init_boundary_exchanges () {
  // The Functor needs to be fully setup to use this function
  assert (is_setup);

  p_->init_boundary_exchanges();
}

template<typename ST>
void EulerStepFunctorST<ST>::precompute_divdp () {
  // The Functor needs to be fully setup to use this function
  assert (is_setup);

  p_->precompute_divdp();
}

template<typename ST>
void EulerStepFunctorST<ST>::
euler_step (const int np1_qdp, const int n0_qdp, const Real dt,
              const Real rhs_multiplier, const DSSOption DSSopt) {
  // The Functor needs to be fully setup to use this function
  assert (is_setup);

  p_->euler_step(np1_qdp, n0_qdp, dt, rhs_multiplier, DSSopt);
}

template<typename ST>
void EulerStepFunctorST<ST>::
qdp_time_avg (const int n0_qdp, const int np1_qdp) {
  // The Functor needs to be fully setup to use this function
  assert (is_setup);

  p_->qdp_time_avg(n0_qdp, np1_qdp);
}

} // namespace Homme

#endif // HOMMEXX_EULER_STEP_FUNCTOR_DEF_HPP
