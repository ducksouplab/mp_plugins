// gstmozzamp/dfm.cpp
#include "dfm.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

static inline void trim(std::string& s) {
  auto wsfront = std::find_if_not(s.begin(), s.end(), [](int c){return std::isspace(c);});
  auto wsback  = std::find_if_not(s.rbegin(), s.rend(), [](int c){return std::isspace(c);}).base();
  if (wsback <= wsfront) { s.clear(); return; }
  s = std::string(wsfront, wsback);
}

std::optional<Deformations> load_dfm(const std::string& path) {
  std::ifstream f(path);
  if (!f) return std::nullopt;

  Deformations d;
  std::string line;
  while (std::getline(f, line)) {
    // drop BOM / CR / comments
    line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
    auto hash = line.find('#'); if (hash != std::string::npos) line.resize(hash);
    trim(line);
    if (line.empty()) continue;

    // tolerate ';' as ',' too
    std::replace(line.begin(), line.end(), ';', ',');

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
