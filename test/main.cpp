#include <iostream>
#include "gtest/gtest.h"

TEST(SomeTest, Test) {
  EXPECT_EQ(0, 0);
}

TEST(SomeTest, Cpp11Feature) {
  auto c = 2;
  auto f = [](int x ) { return 3; };
  f(c);
}

int main(int argc, char** argv) {
  std::cout << "Running tests from gtest\n";
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}