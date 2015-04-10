//
// Created by Arseny Tolmachev on 2015/04/10.
//

#include <kernels.h>
#include <iostream>

#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

namespace ip = boost::interprocess;

int main(int argc, char** argv) {

    ip::file_mapping data(argv[1], ip::read_write);
    ip::mapped_region reg(data, ip::read_write);

    std::cout << "The size is " << reg.get_size() << "\n";
    std::cout << "Hello, world!";
    return 0;
}
