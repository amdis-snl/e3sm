#ifndef PYHOMMEXX_C2F_HPP
#define PYHOMMEXX_C2F_HPP

namespace pyhommexx {

extern "C"
{

void init_parallel_f90 (const int& f_comm);
void prim_init_f90 ();
void prim_forward_f90 ();
void prim_finalize_f90 ();

} // extern "C"

} // namespace pyhommexx

#endif // PYHOMMEXX_C2F_HPP
