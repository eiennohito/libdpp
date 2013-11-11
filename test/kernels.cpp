//
//  kernels.cpp
//  libdpp
//
//  Created by Arseny Tolmachev on 2013-11-11.
//
//

#include "gtest/gtest.h"

#include "kernels.h"

class KernelTest: public ::testing::Test {
  
};

TEST_F(KernelTest, Load) {
  using namespace dpp;
  float data[] = { 1, 2, 6, 2, 4, 1, 6, 1, 9 };

  auto k = l_kernel<float>::from_array(data, 3, 3);
  auto sampler = k->sampler();
  auto items = sampler.sample();
  delete k;
}

TEST_F(KernelTest, Test2) {
  
}
