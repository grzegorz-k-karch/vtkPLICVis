#include "plicvis_impl.h"
#include <cmath>
#include <iostream>
#include <cstdio>
#include <array>
#include <vector>
#include <limits>
#include <iterator>
#include "helper_math.h"
#include "mc_tables.h"
#include "device_launch_parameters.h"

texture<uint, 1, cudaReadModeElementType> edgeTex;
texture<uint, 1, cudaReadModeElementType> triTex;
texture<uint, 1, cudaReadModeElementType> numVertsTex;

void allocateTextures(uint **d_edgeTable, uint **d_triTable,  uint **d_numVertsTable)
{
  cudaChannelFormatDesc channelDesc = 
    cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

  cudaMalloc((void**) d_edgeTable, 256*sizeof(uint));
  cudaMemcpy((void*)*d_edgeTable, (void *)edgeTable, 256*sizeof(uint), cudaMemcpyHostToDevice);
  cudaBindTexture(0, edgeTex, *d_edgeTable, channelDesc);

  cudaMalloc((void**) d_triTable, 256*16*sizeof(uint));
  cudaMemcpy((void*)*d_triTable, (void *)triTable, 256*16*sizeof(uint), cudaMemcpyHostToDevice);
  cudaBindTexture(0, triTex, *d_triTable, channelDesc);

  cudaMalloc((void**) d_numVertsTable, 256*sizeof(uint));
  cudaMemcpy((void*)*d_numVertsTable,(void*)numVertsTable,256*sizeof(uint),cudaMemcpyHostToDevice);
  cudaBindTexture(0, numVertsTex, *d_numVertsTable, channelDesc);
}

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

template <typename T>
cudaArray *copyToCudaArray(const std::vector<T> &data)
{
  const int width = data.size();
  
  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
  cudaArray *cuArray;
  cudaMallocArray(&cuArray, &channelDesc, width);

  // Copy to device memory some data located at address h_data
  // in host memory
  cudaMemcpyToArray(cuArray, 0, 0, data.data(), width*sizeof(T),
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

#define NUMTHREADS 64
#define NUMELEMENTS 8

// ^          ^
// |         /
//  /7-----6
// 3-----2 |
// |  |  | |
// | /4--|-5
// 0-----1/  -->

__device__
float calcLstar(float f_c, float f_nodes[NUMELEMENTS][NUMTHREADS], 
		float3 cellSize, float minVal, float maxVal)
{
  int lidx = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
  float l = 0.0f;

  float isoValue = f_c;
  int iter = 0;
  while (iter < 10) {

    uint cubeindex;
    cubeindex =  uint(f_nodes[0][lidx] < isoValue);
    cubeindex += uint(f_nodes[1][lidx] < isoValue)*2;
    cubeindex += uint(f_nodes[2][lidx] < isoValue)*4;
    cubeindex += uint(f_nodes[3][lidx] < isoValue)*8;
    cubeindex += uint(f_nodes[4][lidx] < isoValue)*16;
    cubeindex += uint(f_nodes[5][lidx] < isoValue)*32;
    cubeindex += uint(f_nodes[6][lidx] < isoValue)*64;
    cubeindex += uint(f_nodes[7][lidx] < isoValue)*128;

    uint numVerts = tex1Dfetch(numVertsTex, cubeindex);

    for (int i = 0; i < numVerts; i+=3) {

      float3 v[3];

      uint edge;
      edge = tex1Dfetch(triTex, (cubeindex*16) + i);
      // v[0] = vertlist[(edge*NTHREADS)+threadIdx.x];
    }

    float volume = 0.0f;//fabs(ComputeVolume(tmpPoly));
    if (volume < f_c)
      maxVal = isoValue;
    else
      minVal = isoValue;
    isoValue = (maxVal+minVal)/2.0f;
    
    iter++;
  }
  return l;
}

__global__
void computeLStar(float *lstar, dim3 roiOffset, dim3 roiSize,
		  cudaTextureObject_t cellGradientsTex,
		  cudaTextureObject_t vofTex,
		  cudaTextureObject_t dxTex,
		  cudaTextureObject_t dyTex,
		  cudaTextureObject_t dzTex)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  int lidx = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;

  __shared__ float f_nodes[NUMELEMENTS][NUMTHREADS];
  
  if (!(i >= roiSize.x ||
	j >= roiSize.y ||
	k >= roiSize.z)) {
 
    float fi = i+0.5f, fj = j+0.5f, fk = k+0.5f;
    // with taylor expansion evaluate f values on nodes
    float f = tex3D<float>(vofTex, fi+roiOffset.x, fj+roiOffset.y, fk+roiOffset.z);
    float3 gf = make_float3(tex3D<float4>(cellGradientsTex, fi, fj, fk));
    float3 cellSize = make_float3(tex1D<float>(dxTex,fi+roiOffset.x),
				  tex1D<float>(dyTex,fj+roiOffset.y),
				  tex1D<float>(dzTex,fk+roiOffset.z));  
    float3 halfCellSize = cellSize/2.0f;  
    float3 dx;
    float minVal = 100000.0f, maxVal = -100000.0f;

    dx = make_float3(-halfCellSize.x,-halfCellSize.y,-halfCellSize.z);
    f_nodes[0][lidx] = f + dot(dx,gf);
    if (minVal > f_nodes[0][lidx]) minVal = f_nodes[0][lidx];
    if (maxVal < f_nodes[0][lidx]) maxVal = f_nodes[0][lidx];

    dx = make_float3( halfCellSize.x,-halfCellSize.y,-halfCellSize.z);
    f_nodes[1][lidx] = f + dot(dx,gf);
    if (minVal > f_nodes[1][lidx]) minVal = f_nodes[1][lidx];
    if (maxVal < f_nodes[1][lidx]) maxVal = f_nodes[1][lidx];

    dx = make_float3( halfCellSize.x, halfCellSize.y,-halfCellSize.z);
    f_nodes[2][lidx] = f + dot(dx,gf);
    if (minVal > f_nodes[2][lidx]) minVal = f_nodes[2][lidx];
    if (maxVal < f_nodes[2][lidx]) maxVal = f_nodes[2][lidx];

    dx = make_float3(-halfCellSize.x, halfCellSize.y,-halfCellSize.z);
    f_nodes[3][lidx] = f + dot(dx,gf);
    if (minVal > f_nodes[3][lidx]) minVal = f_nodes[3][lidx];
    if (maxVal < f_nodes[3][lidx]) maxVal = f_nodes[3][lidx];

    dx = make_float3(-halfCellSize.x,-halfCellSize.y, halfCellSize.z);
    f_nodes[4][lidx] = f + dot(dx,gf);
    if (minVal > f_nodes[4][lidx]) minVal = f_nodes[4][lidx];
    if (maxVal < f_nodes[4][lidx]) maxVal = f_nodes[4][lidx];

    dx = make_float3( halfCellSize.x,-halfCellSize.y, halfCellSize.z);
    f_nodes[5][lidx] = f + dot(dx,gf);
    if (minVal > f_nodes[5][lidx]) minVal = f_nodes[5][lidx];
    if (maxVal < f_nodes[5][lidx]) maxVal = f_nodes[5][lidx];

    dx = make_float3( halfCellSize.x, halfCellSize.y, halfCellSize.z);
    f_nodes[6][lidx] = f + dot(dx,gf);
    if (minVal > f_nodes[6][lidx]) minVal = f_nodes[6][lidx];
    if (maxVal < f_nodes[6][lidx]) maxVal = f_nodes[6][lidx];

    dx = make_float3(-halfCellSize.x, halfCellSize.y, halfCellSize.z);
    f_nodes[7][lidx] = f + dot(dx,gf);
    if (minVal > f_nodes[7][lidx]) minVal = f_nodes[7][lidx];
    if (maxVal < f_nodes[7][lidx]) maxVal = f_nodes[7][lidx];

    float l = calcLstar(f, f_nodes, cellSize, minVal, maxVal);

    int idx = i + j*roiSize.x + k*roiSize.x*roiSize.y;
    lstar[idx] = l;
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

  cudaArray *da_cellGradients = copyToCudaArray3D<float4>(d_cellGradients, 
  							  dim3(roiSize.x, roiSize.y,roiSize.y),
  							  cudaMemcpyDeviceToDevice);
  cudaTextureObject_t cellGradientsTex = createTexture3D(da_cellGradients);

  // allocate textures
  uint *d_edgeTable;
  uint *d_triTable;
  uint *d_numVertsTable;
  allocateTextures(&d_edgeTable, &d_triTable, &d_numVertsTable);

  float *d_lstar;
  cudaMalloc(&d_lstar, numCells*sizeof(float));  
  computeLStar<<<numBlocks, numThreads>>>(d_lstar,roiOffset,roiSize,
  					  cellGradientsTex, vofTex,
  					  dxTex[0], dxTex[1], dxTex[2]);
  // extract plic

  // test
  float *h_lstar = new float[numCells];
  cudaMemcpy(h_lstar, d_lstar, numCells*sizeof(float), cudaMemcpyDeviceToHost);

  int idx = 0;
  for (int k = 0; k < roiSize.z; ++k) {
    for (int j = 0; j < roiSize.y; ++j) {
      for (int i = 0; i < roiSize.x; ++i) {
	if (h_lstar[idx] > 0) {
	  vertices.push_back(make_float4(i,j,k,1.0f));
	  normals.push_back(make_float4(h_lstar[idx]));
	}
	++idx;
      }
    }
  }
  delete [] h_lstar;
  // test end

  cudaFree(d_edgeTable);
  cudaFree(d_triTable);
  cudaFree(d_numVertsTable);

  cudaFree(d_lstar);
  cudaFree(d_cellGradients);
  cudaFree(d_nodeGradients);

  cudaDestroyTextureObject(cellGradientsTex);
  cudaDestroyTextureObject(nodeGradientsTex);
  cudaDestroyTextureObject(vofTex);
  cudaDestroyTextureObject(dxTex[0]);
  cudaDestroyTextureObject(dxTex[1]);
  cudaDestroyTextureObject(dxTex[2]);

  cudaFreeArray(da_cellGradients);
  cudaFreeArray(da_nodeGradients);
  cudaFreeArray(da_vof);
  cudaFreeArray(da_dx[0]);
  cudaFreeArray(da_dx[1]);
  cudaFreeArray(da_dx[2]);
}
