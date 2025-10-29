#include <catch2/catch.hpp>

#include "Types.hpp"
#include "SacadoTypes.hpp"

#include <iostream>
#include <cmath>

TEST_CASE ("Sacado") {

  using RT = Homme::Real;
  using ST = Homme::SFadN<RT,2>;

  // Do some simple Fad ops to ensure we are linking correctly
  RT x0 = 2.0;
  RT y0 = -3.0;

  ST x(x0);
  ST y(y0);

  x.fastAccessDx(0) = 1;
  y.fastAccessDx(1) = 1;

  std::cout << "x: " << x << "\n";
  std::cout << "y: " << y << "\n";

  // f(x,y) = x^2+y^2
  // grad(f) = (2x,2y)

  ST f = x*x + y*y;
  std::cout << "f: " << f << "\n";

  REQUIRE (f.fastAccessDx(0) == 2*x0);
  REQUIRE (f.fastAccessDx(1) == 2*y0);
}
