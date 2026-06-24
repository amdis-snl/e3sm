/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_ADJOINT_UTILS_HPP
#define HOMMEXX_ADJOINT_UTILS_HPP

#include <ekat_assert.hpp>

#include <vector>

namespace Homme
{

// Much like a stack, but we allocate once, and keep track of indices
template<typename T>
struct Tape {
  template<typename... Args>
  Tape(int max_capacity, const Args&... args) {
    EKAT_REQUIRE_MSG(max_capacity > 0, "[Tape::Tape] Error! Invalid capacity.\n");
    capacity = max_capacity;
    tape.resize(capacity, T(args...));
  }

  // Pure state modification: advances the head index
  void shift_fwd() {
    EKAT_REQUIRE_MSG(head + 1 < capacity,
        "[Tape::shift_fwd] Error! Tape is full. Cannot advance.\n");
    ++head;
  }

  // Pure state modification: retrogrades the head index
  void shift_bwd() {
    EKAT_REQUIRE_MSG(head >= 0,
        "[Tape::shift_bwd] Error! Tape head is already at the beginning.\n");
    --head;
  }

  // Pure data access: returns the entry the head is currently resting on
  T& curr() {
    EKAT_REQUIRE_MSG(head >= 0 and head<capacity,
        "[Tape::curr] Error! Tape is empty. No active entry.\n");
    return tape[head];
  }

  const T& curr() const {
    EKAT_REQUIRE_MSG(head >= 0,
        "[Tape::curr] Error! Tape is empty. No active entry.\n");
    return tape[head];
  }

  int capacity = 0;
  int head = -1;
  std::vector<T> tape;
};

} // namespace Homme

#endif // HOMMEXX_ADJOINT_UTILS_HPP
