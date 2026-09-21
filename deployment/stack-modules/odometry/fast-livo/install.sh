#!/bin/bash
# odometry/fast-livo: Sophus (FAST-LIVO2 dep) — strasdat @ a621ff (non-templated,
# double-only; ships libSophus.so + sophus/se3.h). Two Ubuntu-20.04/GCC-9 fixes.
# Ported from pure-jetson-stack Dockerfile `slam` stage.
set -e
git clone https://github.com/strasdat/Sophus.git /opt/Sophus
cd /opt/Sophus && git checkout a621ff
sed -i 's|unit_complex_\.real() = 1\.;|unit_complex_ = std::complex<double>(1., 0.);|' sophus/so2.cpp
sed -i '/unit_complex_\.imag() = 0\.;/d' sophus/so2.cpp
sed -i 's/ -Werror//g' CMakeLists.txt
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null
make -j"$(nproc)" && make install && ldconfig
echo ">> Sophus a621ff baked (libSophus.so + sophus/se3.h)"
