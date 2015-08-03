#include "measure_eigen.hpp"
#include <iostream>

void measureDifferentMatrices();

void measureSizes();

int main(int argc, char** argv) {
  //measureDifferentMatrices();
  measureSizes();
}

void measureSizes() {
  for (int rows = 200; rows < 601; rows += 50) {
    for (int cols = 400; cols < 10001; cols += 200) {
      auto m1 = measure_mult_eig(cols, rows, 5);
      std::cout << rows << " " << cols << " ";
      std::cout << m1.avg << " " << m1.variance << " " << m1.min << " " << m1.max  << "\n";
    }
  }
}

void measureDifferentMatrices() {
  auto m1 = measure_mult_eig(1000, 600, 20);
  std::cout << m1.avg << " (" << m1.variance << ") " << m1.min << " " << m1.max  << "\n";

  auto m2 = measure_svd_jacobi_thinv(1000, 600, 5);
  std::cout << m2.avg << " (" << m2.variance << ") " << m2.min << " " << m2.max  << "\n";

  auto m3 = measure_svd_d_thinv(1000, 600, 20);
  std::cout << m3.avg << " (" << m3.variance << ") " << m3.min << " " << m3.max  << "\n";

  auto m4 = measure_svd_d_thinv(10000, 600, 20);
  std::cout << m4.avg << " (" << m4.variance << ") " << m4.min << " " << m4.max  << "\n";
}