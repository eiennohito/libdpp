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
  float data[] = {1, 2, 3, 4};
  auto k = l_kernel<float>::from_array(data, 2, 2);
}

TEST_F(KernelTest, Test2) {
  
}
