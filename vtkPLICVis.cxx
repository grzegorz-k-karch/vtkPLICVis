#include "vtkObjectFactory.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkSmartPointer.h"
#include "vtkRectilinearGrid.h"
#include "vtkDataArray.h"
#include "vtkFloatArray.h"
#include "vtkCellData.h"
#include "vtkCellArray.h"
#include "vtkPointData.h"
#include "vtkRectilinearGrid.h"

#include "vtkPLICVis.h"
#include "plicvis_impl.h"
#include <vector_types.h>

#include <vector>

vtkStandardNewMacro(vtkPLICVis);

//----------------------------------------------------------------------------
int vtkPLICVis::FillInputPortInformation(int port, vtkInformation* info)
{
  if (!this->Superclass::FillInputPortInformation(port, info)) {
    return 0;
  }
  if (port == 0) {
    info->Set( vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkRectilinearGrid");
    return 1;
  }
  return 0;
}

//----------------------------------------------------------------------------
void nodeCoordsToEdgeLengths(vtkDataArray *coords, const int res,
			     std::vector<float> &dx)
{
  float p0 = coords->GetComponent(0,0);
  for (int i = 0; i < res; ++i) {
    float p1 = coords->GetComponent(i+1,0);
    dx[i] = p1 - p0;
    p0 = p1;
  }
}

//----------------------------------------------------------------------------
void vtkPLICVis::GeneratePLIC(vtkRectilinearGrid *input,
			      vtkPolyData *output)
{
  vtkDataArray *coordsArray[3] = {input->GetXCoordinates(),
				  input->GetYCoordinates(),
				  input->GetZCoordinates()};
  int cellRes[3] = {coordsArray[0]->GetNumberOfTuples()-1,
		    coordsArray[1]->GetNumberOfTuples()-1,
		    coordsArray[2]->GetNumberOfTuples()-1};
  const int numCells = cellRes[0]*cellRes[1]*cellRes[2];
  std::vector<std::vector<float>> dx(3);
  dx[0].resize(cellRes[0]);
  dx[1].resize(cellRes[1]);
  dx[2].resize(cellRes[2]);
  for (int i = 0; i < 3; ++i) {
    nodeCoordsToEdgeLengths(coordsArray[i], cellRes[i], dx[i]);
  }

  vtkDataArray *vofArray = input->GetCellData()->GetArray("Data");
  std::vector<float> vof(numCells);
  for (int i = 0; i < numCells; ++i) {
    vof[i] = vofArray->GetComponent(i,0);
  }

  std::vector<float4> vertices;
  std::vector<int> indices;
  std::vector<float4> normals;

  int extent[6] = {0, cellRes[0]-1, 0, cellRes[1]-1, 0, cellRes[2]-1};

  extractPLIC(vof, cellRes, dx, extent, vertices, indices, normals);

  const int numPoints = vertices.size();
  const int numTriangles = indices.size()/3;

  vtkPoints *points = vtkPoints::New();
  points->SetNumberOfPoints(numPoints);
  for (int i = 0; i < numPoints; ++i) {
    float p[3] = {vertices[i].x, vertices[i].y, vertices[i].z};
    points->SetPoint(i, p);
  }

  vtkIdTypeArray *cellIndices = vtkIdTypeArray::New();
  cellIndices->SetNumberOfComponents(1);
  cellIndices->SetNumberOfTuples(numTriangles*4);
  for (int i = 0; i < numTriangles; ++i) {
    cellIndices->SetValue(i*4, 3);
    cellIndices->SetValue(i*4+1, indices[i*3+0]);
    cellIndices->SetValue(i*4+2, indices[i*3+1]);
    cellIndices->SetValue(i*4+3, indices[i*3+2]);
  }
  vtkCellArray *cells = vtkCellArray::New();
  cells->SetCells(numTriangles, cellIndices);

  vtkFloatArray *normalsArray = vtkFloatArray::New();
  normalsArray->SetNumberOfComponents(3);
  normalsArray->SetNumberOfTuples(numPoints);
  normalsArray->SetName("Normals");
  for (int i = 0; i < numPoints; ++i) {
    float n[3] = {normals[i].x, normals[i].y, normals[i].z};
    normalsArray->SetTuple(i, n);
  }

  output->SetPoints(points);
  output->SetPolys(cells);
  output->GetPointData()->AddArray(normalsArray);
}

//----------------------------------------------------------------------------
int vtkPLICVis::RequestData(vtkInformation *vtkNotUsed(request),
			    vtkInformationVector **inputVector,
			    vtkInformationVector *outputVector)
{
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
  vtkSmartPointer<vtkRectilinearGrid> input = vtkRectilinearGrid::
    SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));

  vtkInformation *outInfo = outputVector->GetInformationObject(0);
  vtkPolyData *output = vtkPolyData::
      SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

  GeneratePLIC(input, output);

  return 1;
}

////////// External Operators /////////////
void vtkPLICVis::PrintSelf(ostream &os, vtkIndent indent)
{
}
