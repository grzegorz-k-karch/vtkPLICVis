#include "plicvis_impl.h"
#include <cmath>
#include <iostream>
#include <cstdio>
#include <array>
#include <vector>
#include <limits>
#include "helper_math.h"
#include "mc_tables.h"
#include "device_launch_parameters.h"


cudaTextureObject_t createTexture1D(cudaArray *array)
{
  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = array;

  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeClamp;
  texDesc.addressMode[1]   = cudaAddressModeClamp;
  texDesc.filterMode       = cudaFilterModePoint;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
  return texObj;
}

cudaTextureObject_t createTexture3D(cudaArray *array)
{
  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = array;

  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeClamp;
  texDesc.addressMode[1]   = cudaAddressModeClamp;
  texDesc.addressMode[2]   = cudaAddressModeClamp;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
  return texObj;
}

cudaArray *copyToCudaArray(const std::vector<float> &data)
{
  const int width = data.size();
  
  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaArray *cuArray;
  cudaMallocArray(&cuArray, &channelDesc, width);

  // Copy to device memory some data located at address h_data
  // in host memory
  cudaMemcpyToArray(cuArray, 0, 0, data.data(), width*sizeof(float),
		    cudaMemcpyHostToDevice);

  return cuArray;
}

template <typename T>
cudaArray *copyToCudaArray3D(const T *data, const dim3 res,
			     cudaMemcpyKind kind = cudaMemcpyHostToDevice)
{
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
  cudaExtent extent = make_cudaExtent(res.x,res.y,res.z);
  cudaArray *cuArray;
  cudaMalloc3DArray(&cuArray, &channelDesc, extent);

  cudaMemcpy3DParms parms[1] = {0};
  parms[0].dstArray = cuArray;
  parms[0].dstPos = make_cudaPos(0,0,0);
  parms[0].extent = extent;
  parms[0].kind = kind;
  parms[0].srcPtr = make_cudaPitchedPtr(const_cast<T*>(data),
					res.x*sizeof(T), res.y, res.z);

  cudaMemcpy3D(parms);

  return cuArray;
}

__global__
void computeNodeGradients(float4 *nodeGradients, dim3 roiOffset, dim3 roiSize,
			  cudaTextureObject_t vofTex,
			  cudaTextureObject_t dxTex,
			  cudaTextureObject_t dyTex,
			  cudaTextureObject_t dzTex)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + roiOffset.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y + roiOffset.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z + roiOffset.z;

  if (!(i >= roiOffset.x + roiSize.x + 1 ||
	j >= roiOffset.y + roiSize.y + 1 ||
	k >= roiOffset.z + roiSize.z + 1)) {

    int im = i-1, jm = j-1, km = k-1;

    float fip = i+0.5f, fjp = j+0.5f, fkp = k+0.5f;
    float fi = i-0.5f, fj = j-0.5f, fk = k-0.5f;

    float dx = tex1D<float>(dxTex, fi);
    float dxp = tex1D<float>(dxTex, fip);
    float dy = tex1D<float>(dyTex, fj);
    float dyp = tex1D<float>(dyTex, fjp);
    float dz = tex1D<float>(dzTex, fk);
    float dzp = tex1D<float>(dzTex, fkp);
    float dc = ((dx+dxp)*(dy+dyp)*(dz+dzp)*0.125f);
        
    float dfm1 = 
      (tex3D<float>(vofTex, fip, fjp, fkp) - 
       tex3D<float>(vofTex, fi, fjp, fkp))*dz + 
      (tex3D<float>(vofTex, fip, fjp, fk) - 
       tex3D<float>(vofTex, fi, fjp, fk))*dzp;
    float dfm2 = 
      (tex3D<float>(vofTex, fip, fj, fkp) - 
       tex3D<float>(vofTex, fi, fj, fkp))*dz + 
      (tex3D<float>(vofTex, fip, fj, fk) - 
       tex3D<float>(vofTex, fi, fj, fk))*dzp;
    float nx = 0.25f*(dfm1*dy+dfm2*dyp) / dc; 

    dfm1 = 
      (tex3D<float>(vofTex, fip, fjp, fkp) - 
       tex3D<float>(vofTex, fip, fj, fkp))*dz + 
      (tex3D<float>(vofTex, fip, fjp, fk) - 
       tex3D<float>(vofTex, fip, fj, fk))*dzp;
    dfm2 = 
      (tex3D<float>(vofTex, fi, fjp, fkp) - 
       tex3D<float>(vofTex, fi, fj, fkp))*dz + 
      (tex3D<float>(vofTex, fi, fjp, fk) - 
       tex3D<float>(vofTex, fi, fj, fk))*dzp;
    float ny = 0.25f*(dfm1*dx+dfm2*dxp) / dc;

    dfm1 = 
      (tex3D<float>(vofTex, fip, fjp, fkp) - 
       tex3D<float>(vofTex, fip, fjp, fk))*dy + 
      (tex3D<float>(vofTex, fip, fj, fkp) - 
       tex3D<float>(vofTex, fip, fj, fk))*dyp;
    dfm2 = 
      (tex3D<float>(vofTex, fi, fjp, fkp) - 
       tex3D<float>(vofTex, fi, fjp, fk))*dy + 
      (tex3D<float>(vofTex, fi, fj, fkp) - 
       tex3D<float>(vofTex, fi, fj, fk))*dyp;
    float nz = 0.25f*(dfm1*dx+dfm2*dxp) / dc;

    i -= roiOffset.x;
    j -= roiOffset.y;
    k -= roiOffset.z;
    int idx = i + j*(roiSize.x+1) + k*(roiSize.x+1)*(roiSize.y+1);
    nodeGradients[idx] = make_float4(-nx, -ny, -nz, 0.0f);
  }
}

__global__
void computeCellGradients(float4 *cellGradients, dim3 roiSize,
			  cudaTextureObject_t nodeGradientsTex)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (!(i >= roiSize.x ||
	j >= roiSize.y ||
	k >= roiSize.z)) {

    float fi = i+0.5f, fj = j+0.5f, fk = k+0.5f;

    float4 gr = make_float4(0.0f);

    gr += tex3D<float4>(nodeGradientsTex, fi,      fj,      fk);
    gr += tex3D<float4>(nodeGradientsTex, fi+1.0f, fj,      fk);
    gr += tex3D<float4>(nodeGradientsTex, fi+1.0f, fj+1.0f, fk);
    gr += tex3D<float4>(nodeGradientsTex, fi,      fj+1.0f, fk);
    gr += tex3D<float4>(nodeGradientsTex, fi,      fj,      fk+1.0f);
    gr += tex3D<float4>(nodeGradientsTex, fi+1.0f, fj,      fk+1.0f);
    gr += tex3D<float4>(nodeGradientsTex, fi+1.0f, fj+1.0f, fk+1.0f);
    gr += tex3D<float4>(nodeGradientsTex, fi,      fj+1.0f, fk+1.0f);

    int idx = i + j*roiSize.x + k*roiSize.x*roiSize.y;
    cellGradients[idx] = gr*0.125f;
  }
}

inline int divUp(int a, int b)
{
  return a/b + (a%b == 0 ? 0 : 1);
}

void extractPLIC(const std::vector<float> &vof, 
		 const int cellRes[3], 
		 const std::vector<std::vector<float>> &dx, 
		 const int extent[6], 
		 std::vector<float4> &vertices, 
		 std::vector<int> &indices, 
		 std::vector<float4> &normals)
{
  cudaArray *da_dx[3] = {copyToCudaArray(dx[0]),
			 copyToCudaArray(dx[1]),
			 copyToCudaArray(dx[2])};
  cudaTextureObject_t dxTex[3] = {createTexture1D(da_dx[0]),
   				  createTexture1D(da_dx[1]),
   				  createTexture1D(da_dx[2])};
  cudaArray *da_vof = copyToCudaArray3D<float>(vof.data(), dim3(cellRes[0],cellRes[1],cellRes[2]));
  cudaTextureObject_t vofTex = createTexture3D(da_vof);

  dim3 roiOffset = dim3(extent[0],extent[2],extent[4]);
  dim3 roiSize = dim3(extent[1]-extent[0]+1,
		      extent[3]-extent[2]+1,
		      extent[5]-extent[4]+1);

  float4 *d_nodeGradients;
  // for node based normals we have to add 1 on each side
  int numNodes = (roiSize.x+1)*(roiSize.y+1)*(roiSize.z+1);  
  cudaMalloc(&d_nodeGradients, numNodes*sizeof(float4));

  dim3 numThreads = dim3(4,4,4);
  // for node based normals we have to add 1 on each side
  dim3 numBlocks = dim3(divUp(roiSize.x+1, numThreads.x),
			divUp(roiSize.y+1, numThreads.y),
			divUp(roiSize.z+1, numThreads.z));
  
  computeNodeGradients<<<numBlocks,numThreads>>>(d_nodeGradients, roiOffset, roiSize,
						 vofTex, dxTex[0], dxTex[1], dxTex[2]);

  cudaArray *da_nodeGradients = copyToCudaArray3D<float4>(d_nodeGradients, 
							  dim3(roiSize.x+1,
							       roiSize.y+1,
							       roiSize.y+1),
							  cudaMemcpyDeviceToDevice);
  cudaTextureObject_t nodeGradientsTex = createTexture3D(da_nodeGradients);
  float4 *d_cellGradients;
  int numCells = roiSize.x*roiSize.y*roiSize.z;  
  cudaMalloc(&d_cellGradients, numCells*sizeof(float4));

  numBlocks = dim3(divUp(roiSize.x, numThreads.x),
		   divUp(roiSize.y, numThreads.y),
		   divUp(roiSize.z, numThreads.z));

  computeCellGradients<<<numBlocks, numThreads>>>(d_cellGradients, roiSize, nodeGradientsTex);


  // compute lstar
  // extract plic

  // test
  float4 *h_cellNormals = new float4[numCells];
  cudaMemcpy(h_cellNormals, d_cellGradients, numCells*sizeof(float4), cudaMemcpyDeviceToHost);

  int idx = 0;
  for (int k = 0; k < roiSize.z; ++k) {
    for (int j = 0; j < roiSize.y; ++j) {
      for (int i = 0; i < roiSize.x; ++i) {
	if (length(make_float3(h_cellNormals[idx])) > 0) {
	  vertices.push_back(make_float4(i,j,k,1.0f));
	  normals.push_back(h_cellNormals[idx]);
	}
	++idx;
      }
    }
  }
  delete [] h_cellNormals;
  // test end

  cudaFree(d_cellGradients);
  cudaFree(d_nodeGradients);

  cudaDestroyTextureObject(nodeGradientsTex);
  cudaDestroyTextureObject(vofTex);
  cudaDestroyTextureObject(dxTex[0]);
  cudaDestroyTextureObject(dxTex[1]);
  cudaDestroyTextureObject(dxTex[2]);
  
  cudaFreeArray(da_nodeGradients);
  cudaFreeArray(da_vof);
  cudaFreeArray(da_dx[0]);
  cudaFreeArray(da_dx[1]);
  cudaFreeArray(da_dx[2]);

  vertices.push_back(make_float4(0,0,0,1));
  vertices.push_back(make_float4(1,0,0,1));
  vertices.push_back(make_float4(0,1,0,1));
  vertices.push_back(make_float4(1,1,0,1));
  indices.push_back(0);
  indices.push_back(1);
  indices.push_back(2);
  indices.push_back(1);
  indices.push_back(3);
  indices.push_back(2);
  normals.push_back(make_float4(0,0,1,0));
  normals.push_back(make_float4(0,0,1,0));
  normals.push_back(make_float4(0,0,1,0));
  normals.push_back(make_float4(0,0,1,0));
}
