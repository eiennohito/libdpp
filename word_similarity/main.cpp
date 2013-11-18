#include "kernels.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cmath>

void print_usage() {
  std::cout << "Usage: word_sim <filename> [<output_file>]\n"
    "\nIf the output file is not provided an <filename>.out is used instead.\n"
    "An input file should have number of dimensions of features in a first line\n"
    "and series of lines following it.\n";
}

class prob_sampler: public dpp::tracer<double> {
  i64 size_;
  std::vector<double> data_;
public:
  void trace(double* data, i64 size, dpp::TraceType tt) override {
    if (tt == dpp::TraceType::ProbabilityDistribution) {
      size_ = size;
      auto pos = data_.size();
      std::copy_n(data, size, std::back_inserter(data_));
      auto sum = std::accumulate(data_.begin() + pos, data_.begin() + pos + size, 0.0, [](double a, double b) { return a + b * b;});
      auto len = std::sqrt(sum);
      std::for_each(data_.begin() + pos, data_.begin() + pos + size, [len](double& item) { item /= len; });
    }
  }
  
  template<typename Stream>
  void print(Stream& s) const {
    i64 rows = data_.size() / size_;
    for (i64 row = 0; row < rows; ++row) {
      for (i64 i = 0; i < size_; ++i) {
        s << data_[row * size_ + i] << " ";
      }
      s << "\n";
    }
  }
};

template<typename R>
std::unique_ptr<R> wrap(R* ptr) { return std::unique_ptr<R>(ptr); }

int main(int argc, char** argv) {
  std::string input_name, output_name;
  
  if (argc == 2) {
    input_name = std::string(argv[1]);
    output_name = input_name + ".out";
  } else if (argc == 3) {
    input_name = std::string(argv[1]);
    output_name = std::string(argv[2]);
  } else {
    print_usage();
    return 1;
  }
  
  std::ifstream input(input_name);
  std::ofstream output(output_name);
  
  if (!input) {
    std::cout << "Invalid input file!\n";
    return 1;
  }
  
  if (!output) {
    std::cout << "Invalid output file!\n";
    return 1;
  }
  
  i64 dims;
  input >> dims;
  
  std::vector<double> features(dims);
  std::vector<i64> indices(dims);
  for (i64 i = 0; i < dims; ++i) {
    indices[i] = i;
  }
  
  std::vector<std::string> words;
  
  auto tracer = std::make_shared<prob_sampler>();
  
  dpp::c_kernel_builder<double> bldr(dims, dims);
  
  bldr.hint_size(100);
  
  while (!input.eof()) {
    std::string word;
    input >> word;
    words.emplace_back(std::move(word));
    for (i64 i = 0; i < dims; ++i) {
      input >> features[i];
    }
    bldr.append(features.data(), indices.data(), dims);
  }
  
  auto kernel = wrap(bldr.build_kernel());
  auto sampler = wrap(kernel->sampler(30));
  sampler->register_tracer(tracer.get());
  
  auto result = sampler->sample();
  
  for(auto pos: result) {
    std::cout << words[pos] << "\n";
  }
  
  tracer->print(output);
  
  return 0;
}