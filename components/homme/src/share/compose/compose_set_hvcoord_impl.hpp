template <typename Sc>
void set_hvcoord (const HommexxReal etai_beg, const HommexxReal etai_end,
                      const Sc* etam) {
  auto& cm = *get_isl_mpi_singleton();
  islmpi::set_hvcoord(cm, etai_beg, etai_end, etam);
}
