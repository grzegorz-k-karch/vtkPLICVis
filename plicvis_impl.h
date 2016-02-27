#ifndef PLICVIS_IMPL_H
#define PLICVIS_IMPL_H

#include <vector>
#include <vector_types.h>

#define EMF0 0.000001f
#define EMF1 0.999999f

void computePLICNormals(const std::vector<float> &vof,
      const int cellRes[3],
			const std::vector<float> &dx,
			const std::vector<float> &dy,
			const std::vector<float> &dz,
			float *normals);

void computePLICPositions(const std::vector<float> &vof,
      const int cellRes[3],
      const std::vector<float> &dx,
      const std::vector<float> &dy,
      const std::vector<float> &dz,
      const float *normals,
      float *positions);

void computePLIC(const std::vector<float> &vof,
      const int cellRes[3],
      const std::vector<float> &dx,
      const std::vector<float> &dy,
      const std::vector<float> &dz,
      const float *normals,
      std::vector<int> &indices,
      std::vector<float> &vertices);

void extractPLIC(const std::vector<float> &vof, 
		 const int cellRes[3], 
		 const std::vector<std::vector<float>> &dx, 
		 const int extent[6], 
		 std::vector<float4> &vertices, 
		 std::vector<int> &indices, 
		 std::vector<float4> &normals);

#endif//PLICVIS_IMPL_H
