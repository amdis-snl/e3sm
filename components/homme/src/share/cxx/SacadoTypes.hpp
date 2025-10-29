#ifndef HOMMEXX_SACADO_TYPES_HPP
#define HOMMEXX_SACADO_TYPES_HPP

#include "Hommexx_config.h"

#include <Sacado_Fad_SFad.hpp>

namespace Homme {

// TODO: decide what to do in terms of which Fad to support (SFad, SLFad, DFad, etc)
template<typename T,int N>
using SFadN = Sacado::Fad::SFad<T,N>;

template<typename T>
using SFad = SFadN<T,HOMMEXX_SFAD_SIZE>;
}

#endif // HOMMEXX_SACADO_TYPES_HPP
