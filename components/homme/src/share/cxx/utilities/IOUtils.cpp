/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "Hommexx_Session.hpp"

#include <iostream>
#include <fstream>

namespace Homme
{

struct OutputRedirection {
  void toggle (bool on) {
    if (on) {
      std::cout.rdbuf(cout);
      std::cerr.rdbuf(cerr);
    } else {
      std::cout.rdbuf(blackhole.rdbuf());
      std::cerr.rdbuf(blackhole.rdbuf());
    }
  }

  static OutputRedirection& instance() {
    static OutputRedirection out_red;
    return out_red;
  }
protected:
  OutputRedirection ()
   : cout (std::cout.rdbuf())
   , cerr (std::cerr.rdbuf())
   , blackhole("/dev/null")
  {}

  std::streambuf* cout;
  std::streambuf* cerr;

  std::ofstream blackhole;
};

void toggle_screen_output (const bool enabled)
{
  OutputRedirection::instance().toggle(enabled);
  Session::m_screen_output_enabled = enabled;
}

} // namespace Homme
