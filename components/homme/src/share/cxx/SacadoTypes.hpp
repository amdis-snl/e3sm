#ifndef HOMMEXX_SACADO_TYPES_HPP
#define HOMMEXX_SACADO_TYPES_HPP

#include "Hommexx_config.h"

#include <Sacado_Fad_SFad.hpp>

namespace Homme {

// TODO: decide what to do in terms of which Fad to support (SFad, SLFad, DFad, etc)
template<typename T>
using SFad = Sacado::Fad::SFad<T,HOMMEXX_SFAD_SIZE>;

}

#endif // HOMMEXX_SACADO_TYPES_HPP
