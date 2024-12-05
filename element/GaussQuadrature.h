/*
  This file is part of TACS: The Toolkit for the Analysis of Composite
  Structures, a parallel finite-element code for structural and
  multidisciplinary design optimization.

  Copyright (C) 2014 Georgia Tech Research Corporation

  TACS is licensed under the Apache License, Version 2.0 (the
  "License"); you may not use this software except in compliance with
  the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
*/

#pragma once

/*
  The following are the definitions of the Gauss (or Gauss-Legendre)
  quadrature points and weights for the interval [-1, 1]. Note that
  these schemes are exact for polynomials of degree 2n - 1.
*/

template <int order>
__HOST_AND_DEVICE__ constexpr double getGaussQuadWeight(const int ptInd) {
  return 0.0;
}

template <int order>
__HOST_AND_DEVICE__ constexpr double getGaussQuadCoord(const int ptInd) {
  return 0.0;
}

// Specializations for the different orders

// 0th order
template <>
__HOST_AND_DEVICE__ constexpr double getGaussQuadWeight<0>(const int ptInd) {
  return 2.0;
}

template <>
__HOST_AND_DEVICE__ constexpr double getGaussQuadCoord<0>(const int ptInd) {
  return 0.0;
}

// 1st order
template <>
__HOST_AND_DEVICE__ constexpr double getGaussQuadWeight<1>(const int ptInd) {
  return 1.0;
}

template <>
__HOST_AND_DEVICE__ constexpr double getGaussQuadCoord<1>(const int ptInd) {
  constexpr double coords[2] = {-0.577350269189626, 0.577350269189626};
  return coords[ptInd];
}

// 2nd order
template <>
__HOST_AND_DEVICE__ constexpr double getGaussQuadWeight<2>(const int ptInd) {
  constexpr double weights[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
  return weights[ptInd];
}

template <>
__HOST_AND_DEVICE__ constexpr double getGaussQuadCoord<2>(const int ptInd) {
  constexpr double coords[3] = {-0.774596669241483, 0.0, 0.774596669241483};
  return coords[ptInd];
}

// 3rd order
template <>
__HOST_AND_DEVICE__ constexpr double getGaussQuadWeight<3>(const int ptInd) {
  constexpr double weights[4] = {0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454};
  return weights[ptInd];
}

template <>
__HOST_AND_DEVICE__ constexpr double getGaussQuadCoord<3>(const int ptInd) {
  constexpr double coords[4] = {-0.861136311594053, -0.339981043584856, 0.339981043584856, 0.861136311594053};
  return coords[ptInd];
}

// 4th order
template <>
__HOST_AND_DEVICE__ constexpr double getGaussQuadWeight<4>(const int ptInd) {
  constexpr double weights[5] = {0.236926885056189,
                                 0.478628670499366,
                                 0.568888888888889,
                                 0.478628670499366,
                                 0.236926885056189};
  return weights[ptInd];
}

template <>
__HOST_AND_DEVICE__ constexpr double getGaussQuadCoord<4>(const int ptInd) {
  constexpr double coords[5] = {-0.906179845938664, -0.538469310105683, 0.0, 0.538469310105683, 0.906179845938664};
  return coords[ptInd];
}

// 5th order
template <>
__HOST_AND_DEVICE__ constexpr double getGaussQuadWeight<5>(const int ptInd) {
  constexpr double weights[6] = {0.1713244923791703450402961,
                                 0.3607615730481386075698335,
                                 0.4679139345726910473898703,
                                 0.4679139345726910473898703,
                                 0.3607615730481386075698335,
                                 0.1713244923791703450402961};
  return weights[ptInd];
}

template <>
__HOST_AND_DEVICE__ constexpr double getGaussQuadCoord<5>(const int ptInd) {
  constexpr double coords[6] = {-0.9324695142031520278123016,
                                -0.6612093864662645136613996,
                                -0.2386191860831969086305017,
                                0.2386191860831969086305017,
                                0.6612093864662645136613996,
                                0.9324695142031520278123016};
  return coords[ptInd];
}

// 6th order
template <>
__HOST_AND_DEVICE__ constexpr double getGaussQuadWeight<6>(const int ptInd) {
  constexpr double weights[7] = {0.1294849661688696932706114,
                                 0.2797053914892766679014678,
                                 0.3818300505051189449503698,
                                 0.4179591836734693877551020,
                                 0.3818300505051189449503698,
                                 0.2797053914892766679014678,
                                 0.1294849661688696932706114};
  return weights[ptInd];
}

template <>
__HOST_AND_DEVICE__ constexpr double getGaussQuadCoord<6>(const int ptInd) {
  constexpr double coords[7] = {-0.9491079123427585245261897,
                                -0.7415311855993944398638648,
                                -0.4058451513773971669066064,
                                0.0,
                                0.4058451513773971669066064,
                                0.7415311855993944398638648,
                                0.9491079123427585245261897};
  return coords[ptInd];
}

// 7th order
template <>
__HOST_AND_DEVICE__ constexpr double getGaussQuadWeight<7>(const int ptInd) {
  constexpr double weights[8] = {0.1012285362903762591525314,
                                 0.2223810344533744705443560,
                                 0.3137066458778872873379622,
                                 0.3626837833783619829651504,
                                 0.3626837833783619829651504,
                                 0.3137066458778872873379622,
                                 0.2223810344533744705443560,
                                 0.1012285362903762591525314};
  return weights[ptInd];
}

template <>
__HOST_AND_DEVICE__ constexpr double getGaussQuadCoord<7>(const int ptInd) {
  constexpr double coords[8] = {-0.9602898564975362316835609,
                                -0.7966664774136267395915539,
                                -0.5255324099163289858177390,
                                -0.1834346424956498049394761,
                                0.1834346424956498049394761,
                                0.5255324099163289858177390,
                                0.7966664774136267395915539,
                                0.9602898564975362316835609};
  return coords[ptInd];
}
