#ifndef HOMMEXX_SACADO_TYPES_HPP
#define HOMMEXX_SACADO_TYPES_HPP

#include "Hommexx_config.h"

#ifdef HOMMEXX_ENABLE_FAD_TYPES

// Disable view specializations
#define SACADO_DISABLE_FAD_VIEW_SPEC

#include <Sacado.hpp>

//#include <Sacado_Fad_SFad.hpp>

namespace Homme {

// TODO: decide what to do in terms of which Fad to support (SFad, SLFad, DFad, etc)
template<typename T,int N>
using SFadN = Sacado::Fad::SFad<T,N>;

template<typename T>
using SFad = SFadN<T,HOMMEXX_SFAD_SIZE>;

using FadType = SFad<double>;

template<typename T, int N>
KOKKOS_INLINE_FUNCTION
T ADValue(const SFadN<T,N>& v) { return v.val(); }

template<typename Expr>
KOKKOS_INLINE_FUNCTION
auto ADValue(const Expr& e)
 -> std::enable_if_t<Sacado::IsExpr<Expr>::value,decltype(e.val())>
{
  return e.val();
}

} // namespace Homme

#endif // HOMMEXX_ENABLE_FAD_TYPES

#endif // HOMMEXX_SACADO_TYPES_HPP
