// dfm.cpp
#include "dfm.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

std::optional<Deformations> load_dfm(const std::string& path) {
  std::ifstream f(path);
  if (!f) return std::nullopt;

  Deformations d;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    if (line[0] == '#') continue;
    for (char& c : line) if (c == ';') c = ','; // tolerate ; as ,
    std::stringstream ss(line);
    DfmEntry e{}; char c;
    if ( (ss >> e.group >> c >> e.idx >> c
            >> e.t0 >> c >> e.t1 >> c >> e.t2 >> c
            >> e.a  >> c >> e.b  >> c >> e.c) ) {
      d.entries.push_back(e);
    }
  }
  return d;
}