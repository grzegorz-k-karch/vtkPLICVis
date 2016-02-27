#ifndef __vtkPLICVis_h
#define __vtkPLICVis_h

#include "vtkPolyDataAlgorithm.h"

class vtkRectilinearGrid;

class vtkPLICVis : public vtkPolyDataAlgorithm
{
 public:
  static vtkPLICVis *New();
  vtkTypeMacro(vtkPLICVis, vtkPolyDataAlgorithm);
  void PrintSelf(ostream &os, vtkIndent indent);

 protected:
  int FillInputPortInformation(int port, vtkInformation* info);
  int RequestData(vtkInformation*, 
		  vtkInformationVector**, 
		  vtkInformationVector*);

 private:
  void GeneratePLIC(vtkRectilinearGrid *,
		    vtkPolyData *);
};
#endif
