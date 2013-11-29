#include "kernels.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

template <typename R>
std::unique_ptr<R> wrap(R *ptr) {
  return std::unique_ptr<R>(ptr);
}

class prob_sampler : public dpp::tracer<double> {
  i64 size_;
  std::vector<double> data_;

 public:
  void trace(double *data, i64 size, dpp::TraceType tt) override {
    if (tt == dpp::TraceType::ProbabilityDistribution) {
      size_ = size;
      auto pos = data_.size();
      std::copy_n(data, size, std::back_inserter(data_));
      auto sum =
          std::accumulate(data_.begin() + pos, data_.begin() + pos + size, 0.0);
      std::for_each(data_.begin() + pos, data_.begin() + pos + size,
                    [sum](double &item) { item /= sum; });
    }
  }

  template <typename Stream>
  void print(Stream &s) const {
    i64 rows = data_.size() / size_;
    for (i64 row = 0; row < rows; ++row) {
      for (i64 i = 0; i < size_; ++i) {
        s << data_[row * size_ + i] << " ";
      }
      s << "\n";
    }
  }
};

struct options {
  i64 items;
  i64 dims;
  std::string input_file;
  std::string trace_file;
  bool greedy_selection = false;
  bool greedy_basis = false;
  bool print_time;
};

std::unique_ptr<options> parse_options(int argc, char **argv) {
  const char *usage =
      "Usage: word_sim <filename>\n"
      "An input file should have number of dimensions of features in "
      "a first line\n"
      "and series of lines following it.\n"
      "\n\nAllowed options";
  po::options_description desc(usage);

  auto &&o = desc.add_options();
  o("help,h", "show this message");
  o("number,n", po::value<i64>()->default_value(30),
    "output a number of items");
  o("trace-file,t", po::value<std::string>(), "output file");
  o("input-file,i", po::value<std::string>(), "input file");
  o("dimension-count,d", po::value<i64>()->default_value(-1),
    "number of dimensions projection. If -1 (default) then equals to number of "
    "dimensions of input file");
  o("greedy-selection,g",
    po::value<bool>()->default_value(false)->implicit_value(true),
    "perfrom a greedy selection from randomly sampled basis");
  // o("greedy-basis,b", po::value<bool>()->default_value(false), "perform a
  // greedy selection of basis (k vectors that have largest corresponding
  // eigenvalues)");

  o("print-time", po::value<bool>()->default_value(false));

  po::positional_options_description p;
  p.add("input-file", 1);

  po::variables_map vm;
  po::store(
      po::command_line_parser(argc, argv).options(desc).positional(p).run(),
      vm);
  po::notify(vm);

  options opts;

  if (vm.count("input-file")) {
    opts.input_file = vm["input-file"].as<std::string>();
  }

  if (vm.count("trace-file")) {
    opts.trace_file = vm["trace-file"].as<std::string>();
  } else {
    opts.trace_file = opts.input_file + ".trace";
  }

  if (vm.count("number")) {
    opts.items = vm["number"].as<i64>();
  }

  if (vm.count("greedy-basis")) {
    opts.greedy_basis = vm["greedy-basis"].as<bool>();
  }

  if (vm.count("greedy-selection")) {
    opts.greedy_selection = vm["greedy-selection"].as<bool>();
  }

  opts.dims = vm["dimension-count"].as<i64>();
  opts.print_time = vm["print-time"].as<bool>();

  if (vm.count("help") || opts.input_file.length() == 0) {
    std::cout << desc;
    return nullptr;
  }

  return std::unique_ptr<options>(new options(std::move(opts)));
}

int main(int argc, char **argv) {

  auto opts = parse_options(argc, argv);

  if (!opts) {
    return 1;
  }

  auto read_begin = std::chrono::high_resolution_clock::now();

  std::ifstream input(opts->input_file);
  std::ofstream output(opts->trace_file);

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

  auto other_dims = opts->dims == -1 ? dims : opts->dims;

  dpp::c_kernel_builder<double> bldr(dims, other_dims);

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

  auto read_end = std::chrono::high_resolution_clock::now();

  auto kernel = wrap(bldr.build_kernel());
  auto kernel_end = std::chrono::high_resolution_clock::now();

  auto sampler = wrap(kernel->sampler(opts->items));
  auto sampler_end = std::chrono::high_resolution_clock::now();

  sampler->register_tracer(tracer.get());

  std::vector<i64> result;

  if (opts->greedy_selection) {
    sampler->greedy(result);
  } else {
    sampler->sample(result);
  }

  auto sample_end = std::chrono::high_resolution_clock::now();

  // auto result = sampler->sample();

  for (auto pos : result) {
    std::cout << words[pos] << "\n";
  }

  tracer->print(output);

  auto d = [](std::chrono::high_resolution_clock::duration dur) {
    return dur.count() /
           static_cast<double>(std::chrono::high_resolution_clock::period::den);
  };

  if (opts->print_time) {
    std::cout << "Read from file in " << d(read_end - read_begin)
              << "\nbuilt kernel in " << d(kernel_end - read_end)
              << "\n built sampler in " << d(sampler_end - kernel_end)
              << "\n created a sample in " << d(sample_end - sampler_end)
              << std::endl;
  }

  return 0;
}