#ifndef PYHOMMEXX_C2F_HPP
#define PYHOMMEXX_C2F_HPP

namespace Homme {
namespace pyhommexx {

extern "C"
{

void init_parallel_f90 (const int& f_comm);
void get_num_unique_pts_f90 (int*& num_per_elem);
void get_unique_pts_f90 (int*& ia_ptr,int*& ja_ptr);
void prim_init_f90 (const char*& filename);
void prim_finalize_f90 ();

} // extern "C"

} // namespace pyhommexx
} // namespace Homme

#endif // PYHOMMEXX_C2F_HPP
