// facewarp_cli: apply basis fields to one image with given landmarks (testing).
//
//   facewarp_cli <image> <landmarks.txt> <basis.json[,basis2.json]> "<AU12=0.5,DOM_o=2>" <out.png> [fold-max]
//
// landmarks.txt: the plugins' LANDMARK_OUTPUT_FILE dump ("Frame ..." header,
// then one "x,y,z" line per landmark, x and y normalised to [0, 1]).
// Prints the folded area and, with fold-max, the safety scale applied.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/imgcodecs.hpp>

#include "../facewarp.hpp"

int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "usage: facewarp_cli <image> <landmarks.txt> <basis[,basis]> <amplitudes> <out.png> [fold-max]\n";
    return 2;
  }
  cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
  if (img.empty()) { std::cerr << "cannot read " << argv[1] << "\n"; return 1; }

  std::vector<cv::Point2f> L;
  std::ifstream lf(argv[2]);
  std::string line;
  while (std::getline(lf, line)) {
    if (line.rfind("Frame", 0) == 0 || line.empty()) continue;
    float x, y, z;
    if (std::sscanf(line.c_str(), "%f,%f,%f", &x, &y, &z) >= 2) L.emplace_back(x * img.cols, y * img.rows);
  }
  if (L.size() < 468) { std::cerr << "need >= 468 landmarks, got " << L.size() << "\n"; return 1; }

  facewarp::Basis basis;
  std::string item, err;
  std::stringstream paths(argv[3]);
  while (std::getline(paths, item, ','))
    if (!basis.load(item, &err)) { std::cerr << err << "\n"; return 1; }

  std::map<std::string, float> amps;
  if (!facewarp::parse_amplitudes(argv[4], &amps, &err)) { std::cerr << err << "\n"; return 1; }
  std::vector<std::string> missing;
  facewarp::Field field = facewarp::combine(basis, amps, &missing);
  for (const auto& m : missing) std::cerr << "warning: no field " << m << " in basis\n";

  facewarp::FaceFrame frame(L);
  std::vector<cv::Point2f> disp(field.size());
  for (size_t i = 0; i < field.size(); ++i) disp[i] = frame.to_px(field[i]);
  facewarp::MeshWarper mw(L, img.cols, img.rows);
  double folded = mw.folded_area(disp);
  float scale = 1.f;
  if (argc > 6) {
    scale = mw.safe_scale(disp, std::atof(argv[6]));
    for (auto& d : disp) d *= scale;
  }
  mw.warp(img, disp);
  cv::imwrite(argv[5], img);
  std::printf("folded_area=%.2f scale=%.3f triangles=%zu\n", folded, scale, mw.tris().size());
  return 0;
}
