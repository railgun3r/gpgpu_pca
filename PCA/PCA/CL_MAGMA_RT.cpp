#include "stdafx.h"
#include "CL_MAGMA_RT.h"

#include <vector>
#include <fstream>
#include <string.h>
#include <cstdint>


using std::string;
using std::ofstream;
using std::ifstream;
using std::ios;
using std::vector;


// define number of command queues to create
#define QUEUE_COUNT 1

#include <io.h> // for _access()
string double_backslashes(string path);

#include <stdlib.h> // for system()

/* 
 * constructor
 */
CL_MAGMA_RT::CL_MAGMA_RT()
{
	HasBeenInitialized = false;

	cpPlatform = NULL;
	ciDeviceCount = 0;
	cdDevices = NULL;
	ceEvent = NULL;
	ckKernel = NULL;
	cxGPUContext = NULL;
	cpPlatform = NULL;
}

/* 
 * destructor 
 */
CL_MAGMA_RT::~CL_MAGMA_RT()
{
	if (!HasBeenInitialized)
		return;

	// Cleanup allocated objects
	if(commandQueue)	delete [] commandQueue;
	if(cdDevices)		magma_free_host(cdDevices); // ACD 2013-07-20 was free(cdDevices), but now it's allocated with magma_malloc_host() = _aligned_malloc()
	if(ceEvent)			clReleaseEvent(ceEvent);  
	if(ckKernel)		clReleaseKernel(ckKernel);  
	if(cxGPUContext)	clReleaseContext(cxGPUContext);
}

cl_command_queue CL_MAGMA_RT::GetCommandQueue(int queueid)
{
	return (queueid>=QUEUE_COUNT)?NULL:commandQueue[queueid];
}

cl_device_id * CL_MAGMA_RT::GetDevicePtr()
{
	return cdDevices;
}

cl_context CL_MAGMA_RT::GetContext()
{
	return cxGPUContext;
}

/*
 * read source code from filename
 * from Rick's clutil
 */
string CL_MAGMA_RT::fileToString(const char* filename)
{
	ifstream fileStream(filename, ios::binary | ios::in | ios::ate);

	if(fileStream.is_open() == true)
	{
		size_t fileSize = fileStream.tellg();

		char* cbuffer = new char[fileSize + 1];

		fileStream.seekg(0, ios::beg);
		fileStream.read(cbuffer, fileSize);
		cbuffer[fileSize] = '\0';

		string  memoryBuffer(cbuffer);
		delete [] cbuffer;
		return memoryBuffer;
	}
	else
	{
		printf ("Error could not open %s\n", filename);
		return NULL;
	}
}


const char* CL_MAGMA_RT::GetErrorCode(cl_int err)
{
		switch(err)
		{
			case CL_SUCCESS:
				return "No Error.";
			case CL_INVALID_MEM_OBJECT:
				return "Invalid memory object.";
			case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
				return "Invalid image format descriptor.";
			case CL_IMAGE_FORMAT_NOT_SUPPORTED:
				return "Image format not supported.";
			case CL_INVALID_IMAGE_SIZE:
				return "Invalid image size.";
			case CL_INVALID_ARG_INDEX:
				return "Invalid argument index for this kernel.";
			case CL_INVALID_ARG_VALUE:
				return "Invalid argument value.";
			case CL_INVALID_SAMPLER:
				return "Invalid sampler.";
			case CL_INVALID_ARG_SIZE:
				return "Invalid argument size.";
			case CL_INVALID_BUFFER_SIZE:
				return "Invalid buffer size.";
			case CL_INVALID_HOST_PTR:
				return "Invalid host pointer.";
			case CL_INVALID_DEVICE:
				return "Invalid device.";
			case CL_INVALID_VALUE:
				return "Invalid value.";
			case CL_INVALID_CONTEXT:
				return "Invalid Context.";
			case CL_INVALID_KERNEL:
				return "Invalid kernel.";
			case CL_INVALID_PROGRAM:
				return "Invalid program object.";
			case CL_INVALID_BINARY:
				return "Invalid program binary.";
			case CL_INVALID_OPERATION:
				return "Invalid operation.";
			case CL_INVALID_BUILD_OPTIONS:
				return "Invalid build options.";
			case CL_INVALID_PROGRAM_EXECUTABLE:
				return "Invalid executable.";
			case CL_INVALID_COMMAND_QUEUE:
				return "Invalid command queue.";
			case CL_INVALID_KERNEL_ARGS:
				return "Invalid kernel arguments.";
			case CL_INVALID_WORK_DIMENSION:
				return "Invalid work dimension.";
			case CL_INVALID_WORK_GROUP_SIZE:
				return "Invalid work group size.";
			case CL_INVALID_WORK_ITEM_SIZE:
				return "Invalid work item size.";
			case CL_INVALID_GLOBAL_OFFSET:
				return "Invalid global offset (should be NULL).";
			case CL_OUT_OF_RESOURCES:
				return "Insufficient resources.";
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:
				return "Could not allocate mem object.";
			case CL_INVALID_EVENT_WAIT_LIST:
				return "Invalid event wait list.";
			case CL_OUT_OF_HOST_MEMORY:
				return "Out of memory on host.";
			case CL_INVALID_KERNEL_NAME:
				return "Invalid kernel name.";
			case CL_INVALID_KERNEL_DEFINITION:
				return "Invalid kernel definition.";
			case CL_BUILD_PROGRAM_FAILURE:
				return "Failed to build program.";
			case CL_MAP_FAILURE:
				return "Failed to map buffer/image";
			case -1001: //This is CL_PLATFORM_NOT_FOUND_KHR
				return "No platforms found. (Did you put ICD files in /etc/OpenCL?)";
			default:
				return "Unknown error.";
		}
}

bool CL_MAGMA_RT::Quit()
{
	if (!HasBeenInitialized)
		return false;

	// Cleanup allocated objects
	if(commandQueue)	delete [] commandQueue;
	if(cdDevices)		magma_free_host(cdDevices); // ACD 2013-07-20 was free(cdDevices), but now it's allocated with magma_malloc_host() = _aligned_malloc()
	if(ceEvent)			clReleaseEvent(ceEvent);  
	if(ckKernel)		clReleaseKernel(ckKernel);  
	if(cxGPUContext)	clReleaseContext(cxGPUContext);

	cpPlatform = NULL;
	ciDeviceCount = 0;
	cdDevices = NULL;
	ceEvent = NULL;
	ckKernel = NULL;
	cxGPUContext = NULL;
	cpPlatform = NULL;

	HasBeenInitialized = false;

	return true;
}

bool CL_MAGMA_RT::Init(cl_platform_id gPlatform, cl_context gContext)
{
  if (HasBeenInitialized)
    {
      printf ("Error: CL_MAGMA_RT has been initialized\n");
      return false;
    }

  printf ("Initializing clMAGMA runtime ...\n");
  
  cl_int ciErrNum = CL_SUCCESS;
  cl_int ciErrNum2 = CL_SUCCESS; 

  // set the platform
  cpPlatform    = gPlatform;

  ciErrNum  = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &ciDeviceCount);
// ACD:
  cdDevices = (cl_device_id *)malloc(ciDeviceCount * sizeof(cl_device_id));
  //cdDevices = (cl_device_id *) _aligned_malloc( (size_t)(ciDeviceCount * sizeof(cl_device_id)), (size_t) 64);
  ciErrNum |= clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, ciDeviceCount, cdDevices, NULL);

printf("=== %d error ===\n", ciErrNum);

  // set the context
  cxGPUContext = gContext;

  printf("=== %d devices ===\n", ciDeviceCount);

   cl_uint addr_data;
   char name_data[64], ext_data[6000];


   for(unsigned int i=0; i<ciDeviceCount; i++) {
   
      ciErrNum= clGetDeviceInfo(cdDevices[i], CL_DEVICE_NAME, 		
            sizeof(name_data), name_data, NULL);			
      if(ciErrNum< 0) {		
         printf("=== %d something wrong ===\n", ciDeviceCount);
      }
      clGetDeviceInfo(cdDevices[i], CL_DEVICE_ADDRESS_BITS, 	
            sizeof(cl_uint), &addr_data, NULL);			

      clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS, 		
            sizeof(ext_data), ext_data, NULL);			

      printf("NAME: %s\nADDRESS_WIDTH: %u\nEXTENSIONS: %s", 
            name_data, addr_data, ext_data);
   }

  // create command-queues                                                                           
  commandQueue = new cl_command_queue[QUEUE_COUNT];
  for(unsigned int i = 0; i < QUEUE_COUNT; i++)
    {
      // create command queue                                                                    
      commandQueue[i] = clCreateCommandQueue(cxGPUContext, cdDevices[0], //TODO: check line 'cdDevices[0]'
					     CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
      if (ciErrNum != CL_SUCCESS)
	{
	  printf (" Error %i in clCreateCommandQueue call !!!\n\n", ciErrNum);
	  return false;
	}
    }

  // setup kernel name -> file name (this will be done later automatically)
  // get directory from environment variable or use default if the clMAGMA_DIR env var isn't set
  // ACD:
  //ACD// const char* dirstr = getenv( "MAGMA_CL_DIR" );
  char* dirstr;
  errno_t err = _dupenv_s( &dirstr, NULL, "clMAGMA_DIR" );
  if( err ) printf( "Error: (clMAGMA_DIR\\interface_opencl\\CL_MAGMA_RT.cpp) _dupenv_s()"); 
  // :ACD
  if ( dirstr == NULL || strlen(dirstr) == 0 ) {
//ACD//  	  dirstr = "/usr/local/magma/cl";
  	  dirstr = "c:\\Program Files\\clMAGMA";
  	  printf( "using default clMAGMA_DIR = %s\n", dirstr );
  }
  // make sure dir path string is terminated with a backslash char
  string dir = dirstr;
  if ( dir.size() > 0 && dir[dir.size()-1] != '\\' ) {
  	  dir += '\\';
  }
  // Point down into the co subdir within clMAGMA_DIR
  dir += "co\\";
  // check for lack of existence of the co directory:  Notify and exit prog if does not exist.
  if( _access( double_backslashes( dir ).c_str(), 0 ) ) { // check for lack of existence of file or dir
	  printf( "ERROR(MAGMA\\interface_opencl\\CL_MAGMA_RT.cpp):\nThe %s directory is missing.\n", dir.c_str() );
//	  system("pause");
	  exit(-1);
  }
  
// ACD 2013-09-17 custom replacement for magma_strsm()
  //Kernel2FileNamePool["strsm_gpu"             ] = dir + "strsm_gpu.cl";
  
  //Kernel2FileNamePool["sinplace_T_even_kernel"] = dir + "sinplace_transpose.cl";
  //Kernel2FileNamePool["sinplace_T_odd_kernel" ] = dir + "sinplace_transpose.cl";
  //Kernel2FileNamePool["stranspose3_32"        ] = dir + "stranspose-v2.cl";
  //Kernel2FileNamePool["stranspose_32"         ] = dir + "stranspose.cl";
  //Kernel2FileNamePool["myslaswp2"             ] = dir + "spermute-v2.cl";

// ACD 2013-09-17 custom replacement for magma_dtrsm()
  //Kernel2FileNamePool["dtrsm_gpu"             ] = dir + "dtrsm_gpu.cl";

  //Kernel2FileNamePool["dinplace_T_even_kernel"] = dir + "dinplace_transpose.cl";
  //Kernel2FileNamePool["dinplace_T_odd_kernel" ] = dir + "dinplace_transpose.cl";
  //Kernel2FileNamePool["dtranspose3_32"        ] = dir + "dtranspose-v2.cl";
  //Kernel2FileNamePool["dtranspose_32"         ] = dir + "dtranspose.cl";
  //Kernel2FileNamePool["mydlaswp2"             ] = dir + "dpermute-v2.cl";

// ACD 2013-09-17 custom replacement for magma_ctrsm()
  //Kernel2FileNamePool["ctrsm_gpu"             ] = dir + "ctrsm_gpu.cl";

  //Kernel2FileNamePool["cinplace_T_even_kernel"] = dir + "cinplace_transpose.cl";
  //Kernel2FileNamePool["cinplace_T_odd_kernel" ] = dir + "cinplace_transpose.cl";
  //Kernel2FileNamePool["ctranspose3_32"        ] = dir + "ctranspose-v2.cl";
  //Kernel2FileNamePool["ctranspose_32"         ] = dir + "ctranspose.cl";
  //Kernel2FileNamePool["myclaswp2"             ] = dir + "cpermute-v2.cl";

// ACD 2013-09-17 custom replacement for magma_ztrsm()
  //Kernel2FileNamePool["ztrsm_gpu"             ] = dir + "ztrsm_gpu.cl";

  //Kernel2FileNamePool["zinplace_T_even_kernel"] = dir + "zinplace_transpose.cl";
  //Kernel2FileNamePool["zinplace_T_odd_kernel" ] = dir + "zinplace_transpose.cl";
  //Kernel2FileNamePool["ztranspose3_32"        ] = dir + "ztranspose-v2.cl";
  //Kernel2FileNamePool["ztranspose_32"         ] = dir + "ztranspose.cl";
  //Kernel2FileNamePool["myzlaswp2"             ] = dir + "zpermute-v2.cl";

//auxiliary functions
  //Kernel2FileNamePool["sset_nbxnb_to_zero"    ] = dir + "sauxiliary.cl";
  //Kernel2FileNamePool["dset_nbxnb_to_zero"    ] = dir + "dauxiliary.cl";
  //Kernel2FileNamePool["cset_nbxnb_to_zero"    ] = dir + "cauxiliary.cl";
  //Kernel2FileNamePool["zset_nbxnb_to_zero"    ] = dir + "zauxiliary.cl";
  //Kernel2FileNamePool["slaset"    ] = dir + "sauxiliary.cl";
  //Kernel2FileNamePool["dlaset"    ] = dir + "dauxiliary.cl";
  //Kernel2FileNamePool["claset"    ] = dir + "cauxiliary.cl";
  //Kernel2FileNamePool["zlaset"    ] = dir + "zauxiliary.cl";
  //Kernel2FileNamePool["slaset_lower"    ] = dir + "sauxiliary.cl";
  //Kernel2FileNamePool["dlaset_lower"    ] = dir + "dauxiliary.cl";
  //Kernel2FileNamePool["claset_lower"    ] = dir + "cauxiliary.cl";
  //Kernel2FileNamePool["zlaset_lower"    ] = dir + "zauxiliary.cl";
  //Kernel2FileNamePool["slaset_upper"    ] = dir + "sauxiliary.cl";
  //Kernel2FileNamePool["dlaset_upper"    ] = dir + "dauxiliary.cl";
  //Kernel2FileNamePool["claset_upper"    ] = dir + "cauxiliary.cl";
  //Kernel2FileNamePool["zlaset_upper"    ] = dir + "zauxiliary.cl";

//zlacpy functions
  //Kernel2FileNamePool["slacpy_kernel"    ] = dir + "slacpy.cl";
  //Kernel2FileNamePool["dlacpy_kernel"    ] = dir + "dlacpy.cl";
  //Kernel2FileNamePool["clacpy_kernel"    ] = dir + "clacpy.cl";
  //Kernel2FileNamePool["zlacpy_kernel"    ] = dir + "zlacpy.cl";

//zswap functions
  //Kernel2FileNamePool["magmagpu_sswap"    ] = dir + "sswap.cl";
  //Kernel2FileNamePool["magmagpu_dswap"    ] = dir + "dswap.cl";
  //Kernel2FileNamePool["magmagpu_cswap"    ] = dir + "cswap.cl";
  //Kernel2FileNamePool["magmagpu_zswap"    ] = dir + "zswap.cl";

  HasBeenInitialized = true;

// ACD 2013-09-17 magma_strsm() replacement
  //BuildFromSources((dir + "strsm_gpu.cl").c_str());

  //BuildFromSources((dir + "sinplace_transpose.cl").c_str()); //TODO: corrupted
  //BuildFromSources((dir + "stranspose-v2.cl").c_str());
  //BuildFromSources((dir + "stranspose.cl").c_str());
  //BuildFromSources((dir + "spermute-v2.cl").c_str());

// ACD 2013-09-17 magma_dtrsm() replacement
  //BuildFromSources((dir + "dtrsm_gpu.cl").c_str());

  //BuildFromSources((dir + "dinplace_transpose.cl").c_str()); //TODO: corrupted
  //BuildFromSources((dir + "dtranspose-v2.cl").c_str());
  //BuildFromSources((dir + "dtranspose.cl").c_str());
  //BuildFromSources((dir + "dpermute-v2.cl").c_str());

// ACD 2013-09-17 magma_ctrsm() replacement
  //BuildFromSources((dir + "ctrsm_gpu.cl").c_str());

  //BuildFromSources((dir + "cinplace_transpose.cl").c_str()); //TODO: corrupted
  //BuildFromSources((dir + "ctranspose-v2.cl").c_str());
  //BuildFromSources((dir + "ctranspose.cl").c_str());
  //BuildFromSources((dir + "cpermute-v2.cl").c_str());

// ACD 2013-09-17 magma_ztrsm() replacement
  //BuildFromSources((dir + "ztrsm_gpu.cl").c_str());

  //BuildFromSources((dir + "zinplace_transpose.cl").c_str()); //TODO: corrupted
  //BuildFromSources((dir + "ztranspose-v2.cl").c_str());
  //BuildFromSources((dir + "ztranspose.cl").c_str());
  //BuildFromSources((dir + "zpermute-v2.cl").c_str());

  //BuildFromSources((dir + "sauxiliary.cl").c_str());
  //BuildFromSources((dir + "dauxiliary.cl").c_str());
  //BuildFromSources((dir + "cauxiliary.cl").c_str());
  //BuildFromSources((dir + "zauxiliary.cl").c_str());
 
  //BuildFromSources((dir + "slacpy.cl").c_str());
  //BuildFromSources((dir + "dlacpy.cl").c_str());
  //BuildFromSources((dir + "clacpy.cl").c_str());
  //BuildFromSources((dir + "zlacpy.cl").c_str());

  //BuildFromSources((dir + "sswap.cl").c_str());
  //BuildFromSources((dir + "dswap.cl").c_str());
  //BuildFromSources((dir + "cswap.cl").c_str());
  //BuildFromSources((dir + "zswap.cl").c_str());

  bool rtr;
  
  
// ACD 2013-09-17 magma_strsm() replacement

  //rtr = CreateKernel("strsm_gpu");
  //if (rtr==false)
  //printf ("error creating kernel strsm_gpu\n");
  //rtr = CreateKernel("strsm_gpu");
 
  //rtr = CreateKernel("sinplace_T_even_kernel");
  //if (rtr==false)
  //  printf ("error creating kernel sinplace_T_even_kernel\n");
  //rtr = CreateKernel("sinplace_T_odd_kernel");
  //if (rtr==false)
  //  printf ("error creating kernel sinplace_T_odd_kernel\n");
  //rtr = CreateKernel("stranspose3_32");
  //if (rtr==false)
  //  printf ("error creating kernel stranspose3_32\n");
  //rtr = CreateKernel("stranspose_32");
  //if (rtr==false)
  //  printf ("error creating kernel stranspose_32\n");
  //rtr = CreateKernel("myslaswp2");
  //if (rtr==false)
  //  printf ("error creating kernel myslaswp2\n");

// ACD 2013-09-17 magma_dtrsm() replacement

  //rtr = CreateKernel("dtrsm_gpu");
  //if (rtr==false)
  //  printf ("error creating kernel dtrsm_gpu\n");
  //rtr = CreateKernel("dtrsm_gpu");

  //rtr = CreateKernel("dinplace_T_even_kernel");
  //if (rtr==false)
  //  printf ("error creating kernel dinplace_T_even_kernel\n");
  //rtr = CreateKernel("dinplace_T_odd_kernel");
  //if (rtr==false)
  //  printf ("error creating kernel dinplace_T_odd_kernel\n");
  //rtr = CreateKernel("dtranspose3_32");
  //if (rtr==false)
  //  printf ("error creating kernel dtranspose3_32\n");
  //rtr = CreateKernel("dtranspose_32");
  //if (rtr==false)
  //  printf ("error creating kernel dtranspose_32\n");
  //rtr = CreateKernel("mydlaswp2");
  //if (rtr==false)
  //  printf ("error creating kernel mydlaswp2\n");

// ACD 2013-09-17 magma_ctrsm() replacement
  //rtr = CreateKernel("ctrsm_gpu");
  //if (rtr==false)
  //  printf ("error creating kernel ctrsm_gpu\n");
  //rtr = CreateKernel("ctrsm_gpu");

  //rtr = CreateKernel("cinplace_T_even_kernel");
  //if (rtr==false)
  //  printf ("error creating kernel cinplace_T_even_kernel\n");
  //rtr = CreateKernel("cinplace_T_odd_kernel");
  //if (rtr==false)
  //  printf ("error creating kernel cinplace_T_odd_kernel\n");
  //rtr = CreateKernel("ctranspose3_32");
  //if (rtr==false)
  //  printf ("error creating kernel ctranspose3_32\n");
  //rtr = CreateKernel("ctranspose_32");
  //if (rtr==false)
  //  printf ("error creating kernel ctranspose_32\n");
  //rtr = CreateKernel("myclaswp2");
  //if (rtr==false)
  //  printf ("error creating kernel myclaswp2\n");

// ACD 2013-09-17 magma_ztrsm() replacement
  //rtr = CreateKernel("ztrsm_gpu");
  //if (rtr==false)
  //  printf ("error creating kernel ztrsm_gpu\n");
  //rtr = CreateKernel("ztrsm_gpu");
 
  //rtr = CreateKernel("zinplace_T_even_kernel");
  //if (rtr==false)
  //  printf ("error creating kernel zinplace_T_even_kernel\n");
  //rtr = CreateKernel("zinplace_T_odd_kernel");
  //if (rtr==false)
  //  printf ("error creating kernel zinplace_T_odd_kernel\n");
  //rtr = CreateKernel("ztranspose3_32");
  //if (rtr==false)
  //  printf ("error creating kernel ztranspose3_32\n");
  //rtr = CreateKernel("ztranspose_32");
  //if (rtr==false)
  //  printf ("error creating kernel ztranspose_32\n");
  //rtr = CreateKernel("myzlaswp2");
  //if (rtr==false)
  //  printf ("error creating kernel myzlaswp2\n");

  //rtr = CreateKernel("sset_nbxnb_to_zero");
  //if (rtr==false)
  //  printf ("error creating kernel sset_nbxnb_zero\n");
  //rtr = CreateKernel("dset_nbxnb_to_zero");
  //if (rtr==false)
  //  printf ("error creating kernel dset_nbxnb_zero\n");
  //rtr = CreateKernel("cset_nbxnb_to_zero");
  //if (rtr==false)
  //  printf ("error creating kernel cset_nbxnb_zero\n");
  //rtr = CreateKernel("zset_nbxnb_to_zero");
  //if (rtr==false)
  //  printf ("error creating kernel zset_nbxnb_zero\n");
  //rtr = CreateKernel("slaset");
  //if (rtr==false)
  //  printf ("error creating kernel slaset\n");
  //rtr = CreateKernel("dlaset");
  //if (rtr==false)
  //  printf ("error creating kernel dlaset\n");
  //rtr = CreateKernel("claset");
  //if (rtr==false)
  //  printf ("error creating kernel claset");
  //rtr = CreateKernel("zlaset");
  //if (rtr==false)
  //  printf ("error creating kernel zlaset\n");
  //rtr = CreateKernel("slaset_lower");
  //if (rtr==false)
  //  printf ("error creating kernel slaset_lower\n");
  //rtr = CreateKernel("dlaset_lower");
  //if (rtr==false)
  //  printf ("error creating kernel dlaset_lower\n");
  //rtr = CreateKernel("claset_lower");
  //if (rtr==false)
  //  printf ("error creating kernel claset_lower");
  //rtr = CreateKernel("zlaset_lower");
  //if (rtr==false)
  //  printf ("error creating kernel zlaset_lower\n");
  //rtr = CreateKernel("slaset_upper");
  //if (rtr==false)
  //  printf ("error creating kernel slaset_upper\n");
  //rtr = CreateKernel("dlaset_upper");
  //if (rtr==false)
  //  printf ("error creating kernel dlaset_upper\n");
  //rtr = CreateKernel("claset_upper");
  //if (rtr==false)
  //  printf ("error creating kernel claset_upper");
  //rtr = CreateKernel("zlaset_upper");
  //if (rtr==false)
  //  printf ("error creating kernel zlaset_upper\n");
 
  //rtr = CreateKernel("slacpy_kernel");
  //if (rtr==false)
	 // printf ("error creating kernel slacpy_kernel\n");
  //rtr = CreateKernel("dlacpy_kernel");
  //if (rtr==false)
	 // printf ("error creating kernel dlacpy_kernel\n");
  //rtr = CreateKernel("clacpy_kernel");
  //if (rtr==false)
	 // printf ("error creating kernel clacpy_kernel");
  //rtr = CreateKernel("zlacpy_kernel");
  //if (rtr==false)
	 // printf ("error creating kernel zlacpy_kernel\n");

  //rtr = CreateKernel("magmagpu_sswap");
  //if (rtr==false)
	 // printf ("error creating kernel magmagpu_sswap\n");
  //rtr = CreateKernel("magmagpu_dswap");
  //if (rtr==false)
	 // printf ("error creating kernel magmagpu_dswap\n");
  //rtr = CreateKernel("magmagpu_cswap");
  //if (rtr==false)
	 // printf ("error creating kernel magmagpu_cswap\n");
  //rtr = CreateKernel("magmagpu_zswap");
  //if (rtr==false)
	 // printf ("error creating kernel magmagpu_zswap\n");

  return true;
}


bool CL_MAGMA_RT::Init()
{
	if (HasBeenInitialized)
	{
		printf ("Error: CL_MAGMA_RT has been initialized\n");
		return false;
	}

	printf ("Initializing...\n");

	/*
	 * initialize OpenCL runtime
	 */
	cl_int ciErrNum = CL_SUCCESS;
	
	// Get the platform
	cl_uint ione = 1;
	ciErrNum = clGetPlatformIDs(1, &cpPlatform, &ione);
	if (ciErrNum != CL_SUCCESS)
	{
		printf("Error: Failed to create OpenCL context!\n");
//		return ciErrNum;// ACD 2013-01-23c
		return false; // ACD 2013-01-23c  This subr is of type bool, so return value has to be T or F.
	}

	ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &ciDeviceCount);
// ACD:
	cdDevices = (cl_device_id *)malloc(ciDeviceCount * sizeof(cl_device_id));
	//cdDevices = (cl_device_id *) _aligned_malloc( (size_t)(ciDeviceCount * sizeof(cl_device_id)), (size_t) 64);
	ciErrNum |= clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, ciDeviceCount, cdDevices, NULL);
	if (ciErrNum != CL_SUCCESS)
	{
		printf("Error: clGetDeviceIDs at %d in file %s!\n", __LINE__, __FILE__);
		return false;
	}

	//Create the context
	cxGPUContext = clCreateContext(0, ciDeviceCount, cdDevices, NULL, NULL, &ciErrNum);
	if (ciErrNum != CL_SUCCESS)
	{
		printf("Error: Failed to create OpenCL context!\n");
		return false;
	}   
		
		/*
	// Find out how many GPU's to compute on all available GPUs
	size_t nDeviceBytes;
	ciErrNum = clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &nDeviceBytes);
	if (ciErrNum != CL_SUCCESS)
	{   
		printf (" Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
		return ciErrNum;
	}
	else if (ciDeviceCount == 0)
	{
		printf (" There are no devices supporting OpenCL (return code %i)\n\n", ciErrNum);
		return false;
	}
	ciDeviceCount = (cl_uint)nDeviceBytes/sizeof(cl_device_id); 
	*/

	// show device 
	for(unsigned int i = 0; i < ciDeviceCount; i++)
	{
		// get and print the device for this queue
		//cl_device_id device = oclGetDev(cxGPUContext, i);

		char deviceName[1024];
		memset(deviceName, '\0', 1024);
		clGetDeviceInfo(cdDevices[i], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
		printf ("Device: %s\n", deviceName);
	}

	// create command-queues
	commandQueue = new cl_command_queue[QUEUE_COUNT];
	for(unsigned int i = 0; i < QUEUE_COUNT; i++)
	{
		// create command queue
		commandQueue[i] = clCreateCommandQueue(cxGPUContext, cdDevices[0], CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
		if (ciErrNum != CL_SUCCESS)
		{
			printf (" Error %i in clCreateCommandQueue call !!!\n\n", ciErrNum);
			return false;
		}
	}

  // get directory from environment variable or use default if the clMAGMA_DIR env var isn't set
  // ACD:
  //ACD// const char* dirstr = getenv( "MAGMA_CL_DIR" );
  char* dirstr;
  errno_t err = _dupenv_s( &dirstr, NULL, "clMAGMA_DIR" );
  if( err ) printf( "Error: (clMAGMA_DIR\\interface_opencl\\CL_MAGMA_RT.cpp) _dupenv_s()"); 
  // :ACD
  if ( dirstr == NULL || strlen(dirstr) == 0 ) {
//ACD//  	  dirstr = "/usr/local/magma/cl";
  	  dirstr = "c:\\Program Files\\clMAGMA";
  	  printf( "using default clMAGMA_DIR = %s\n", dirstr );
  }
  // make sure dir path string is terminated with a backslash char
  string dir = dirstr;
  if ( dir.size() > 0 && dir[dir.size()-1] != '\\' ) {
  	  dir += '\\';
  }
  // Point down into the co subdir within clMAGMA_DIR
  dir += "co\\";
  // check for lack of existence of the co directory:  Notify and exit prog if does not exist.
  if( _access( double_backslashes( dir ).c_str(), 0 ) ) { // check for lack of existence of file or dir
	  printf( "ERROR(MAGMA\\interface_opencl\\CL_MAGMA_RT.cpp):\nThe %s directory is missing.\n", dir.c_str() );
//	  system("pause");
	  exit(-1);
  }

  // setup kernel name -> file name (this will be done later automatically)
//    string dir = "/Users/mgates/Documents/magma-cl/interface_opencl/";

// ACD 2013-09-14/15:  It seems this section of code may not be used by clMAGMA.

// ACD 2013-09-16 custom replacement for magma_strsm()
    //Kernel2FileNamePool["strsm_gpu"             ]  = dir + string("strsm_gpu.cl");
  
	//Kernel2FileNamePool["sinplace_T_even_kernel"] = dir + string("sinplace_transpose.cl");
	//Kernel2FileNamePool["sinplace_T_odd_kernel" ] = dir + string("sinplace_transpose.cl");
	//Kernel2FileNamePool["stranspose3_32"        ] = dir + string("stranspose-v2.cl");
	//Kernel2FileNamePool["stranspose_32"         ] = dir + string("stranspose.cl");
	//Kernel2FileNamePool["myslaswp2"             ] = dir + string("spermute-v2.cl");

// ACD 2013-09-16 custom replacement for magma_dtrsm()
    //Kernel2FileNamePool["dtrsm_gpu"             ]  = dir + string("dtrsm_gpu.cl");
  
	//Kernel2FileNamePool["dinplace_T_even_kernel"] = dir + string("dinplace_transpose.cl");
	//Kernel2FileNamePool["dinplace_T_odd_kernel" ] = dir + string("dinplace_transpose.cl");
	//Kernel2FileNamePool["dtranspose3_32"        ] = dir + string("dtranspose-v2.cl");
	//Kernel2FileNamePool["dtranspose_32"         ] = dir + string("dtranspose.cl");
	//Kernel2FileNamePool["mydlaswp2"             ] = dir + string("dpermute-v2.cl");

// ACD 2013-09-16 custom replacement for magma_ctrsm()
    //Kernel2FileNamePool["ctrsm_gpu"             ]  = dir + string("ctrsm_gpu.cl");
  
	//Kernel2FileNamePool["cinplace_T_even_kernel"] = dir + string("cinplace_transpose.cl");
	//Kernel2FileNamePool["cinplace_T_odd_kernel" ] = dir + string("cinplace_transpose.cl");
	//Kernel2FileNamePool["ctranspose3_32"        ] = dir + string("ctranspose-v2.cl");
	//Kernel2FileNamePool["ctranspose_32"         ] = dir + string("ctranspose.cl");
	//Kernel2FileNamePool["myclaswp2"             ] = dir + string("cpermute-v2.cl");

// ACD 2013-09-16 custom replacement for magma_ztrsm()
    //Kernel2FileNamePool["ztrsm_gpu"             ]  = dir + string("ztrsm_gpu.cl");
  
	//Kernel2FileNamePool["zinplace_T_even_kernel"] = dir + string("zinplace_transpose.cl");
	//Kernel2FileNamePool["zinplace_T_odd_kernel" ] = dir + string("zinplace_transpose.cl");
	//Kernel2FileNamePool["ztranspose3_32"        ] = dir + string("ztranspose-v2.cl");
	//Kernel2FileNamePool["ztranspose_32"         ] = dir + string("ztranspose.cl");
	//Kernel2FileNamePool["myzlaswp2"             ] = dir + string("zpermute-v2.cl");

	//auxiliary functions
	//Kernel2FileNamePool["sset_nbxnb_to_zero"    ] = dir + string("sauxiliary.cl");
	//Kernel2FileNamePool["dset_nbxnb_to_zero"    ] = dir + string("dauxiliary.cl");
	//Kernel2FileNamePool["cset_nbxnb_to_zero"    ] = dir + string("cauxiliary.cl");
	//Kernel2FileNamePool["zset_nbxnb_to_zero"    ] = dir + string("zauxiliary.cl");
	//Kernel2FileNamePool["slaset"    ] = dir + string("sauxiliary.cl");
	//Kernel2FileNamePool["dlaset"    ] = dir + string("dauxiliary.cl");
	//Kernel2FileNamePool["claset"    ] = dir + string("cauxiliary.cl");
	//Kernel2FileNamePool["zlaset"    ] = dir + string("zauxiliary.cl");
	//Kernel2FileNamePool["slaset_lower"    ] = dir + string("sauxiliary.cl");
	//Kernel2FileNamePool["dlaset_lower"    ] = dir + string("dauxiliary.cl");
	//Kernel2FileNamePool["claset_lower"    ] = dir + string("cauxiliary.cl");
	//Kernel2FileNamePool["zlaset_lower"    ] = dir + string("zauxiliary.cl");
	//Kernel2FileNamePool["slaset_upper"    ] = dir + string("sauxiliary.cl");
	//Kernel2FileNamePool["dlaset_upper"    ] = dir + string("dauxiliary.cl");
	//Kernel2FileNamePool["claset_upper"    ] = dir + string("cauxiliary.cl");
	//Kernel2FileNamePool["zlaset_upper"    ] = dir + string("zauxiliary.cl");
    
	//zlacpy functions
	//Kernel2FileNamePool["slacpy_kernel"    ] = dir + string("slacpy.cl");
	//Kernel2FileNamePool["dlacpy_kernel"    ] = dir + string("dlacpy.cl");
	//Kernel2FileNamePool["clacpy_kernel"    ] = dir + string("clacpy.cl");
	//Kernel2FileNamePool["zlacpy_kernel"    ] = dir + string("zlacpy.cl");

	//zswap functions
	//Kernel2FileNamePool["magmagpu_sswap"    ] = dir + string("sswap.cl");
	//Kernel2FileNamePool["magmagpu_dswap"    ] = dir + string("dswap.cl");
	//Kernel2FileNamePool["magmagpu_cswap"    ] = dir + string("cswap.cl");
	//Kernel2FileNamePool["magmagpu_zswap"    ] = dir + string("zswap.cl");

	HasBeenInitialized = true;
	return true;
}


int CL_MAGMA_RT::GatherFilesToCompile( const char* FileNameList, vector<string>& FileNames)
{
	if (FileNameList==NULL || strlen(FileNameList)==0)
		return -1;

	ifstream fileStream(FileNameList, ifstream::in);
	
	int num=0;
	if(fileStream.is_open())
	{
		while (!fileStream.eof())
		{
			char buff[512];

			fileStream.getline (buff,512);
			
			if (strlen(buff) && buff[0]!='#')
			{
				FileNames.push_back (string(buff));
				memset (buff, ' ', 512);
				num++;
			}
		}

	}
	fileStream.close();

	return num;
}

/*
 * this function build .cl files and store the bits to .o files
 */
bool CL_MAGMA_RT::CompileSourceFiles( const char* FileNameList )
{
	if (FileNameList==NULL)
		return false;

	//read from clfile for a list of cl files to compile  
	vector<string> FileNames;
	int NumOfFiles = GatherFilesToCompile (FileNameList, FileNames);

	if (NumOfFiles==0)
		return false;

	//compile each cl file
	vector<string>::iterator it;
	for (it=FileNames.begin(); it<FileNames.end(); it++ )
	{ 
		printf ("compiling %s\n", it->c_str());
		bool ret = CompileFile (it->c_str()); 
		if (ret==false)
		{
			printf ("Error while trying to compile %s\n", it->c_str());
			return false;
		}
	}

	return true;
}

bool CL_MAGMA_RT::CompileFile(const char *FileName)
{
	if (FileName==NULL)
	{
		printf ("Error: file name empty on line %d in %s\n", __LINE__, __FILE__);
		return false;
	}

	if (!HasBeenInitialized)
		Init();

	// read in the kernel source
	string fileStrings;

	fileStrings = fileToString(FileName);
	const char *filePointers = fileStrings.c_str();

	// Create the program
	cl_program cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char**)&filePointers, NULL, &ciErrNum);
	if (ciErrNum != CL_SUCCESS)
	{
		printf ("Error: clCreateProgramWithSource at %d in %s\n", __LINE__, __FILE__);
		return false;
	}
	
	// Build the program
	// MUST do this otherwise clGetProgramInfo return zeros for binary sizes
	ciErrNum = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
	if (ciErrNum != CL_SUCCESS)
	{
		printf ("clBuildProgram error at %d in %s : ", __LINE__, __FILE__);
		switch( ciErrNum ) {
			case CL_INVALID_PROGRAM:
				printf("CL_INVALID_PROGRAM\n");
				break;
				
			case CL_INVALID_VALUE:
				printf("CL_INVALID_VALUE\n");
				break;
				
			case CL_INVALID_DEVICE:
				printf("CL_INVALID_DEVICE\n");
				break;
				
			case CL_INVALID_BINARY:
				printf("CL_INVALID_BINARY\n");
				break;
				
			case CL_INVALID_BUILD_OPTIONS:
				printf("CL_INVALID_BUILD_OPTIONS\n");
				break;
				
			case CL_INVALID_OPERATION:
				printf("CL_INVALID_OPERATION\n");
				break;
				
			case CL_COMPILER_NOT_AVAILABLE:
				printf("CL_COMPILER_NOT_AVAILABLE\n");
				break;
				
			case CL_BUILD_PROGRAM_FAILURE:
				printf("CL_BUILD_PROGRAM_FAILURE\n");
				break;
				
			case CL_OUT_OF_HOST_MEMORY:
				printf("CL_OUT_OF_HOST_MEMORY\n");
				break;
				
			default:
				printf(" Unknown error.\n");
				break;
		}
		
		printf("NOTE ERROR:  Enter any non-space text to continue.\n");
		char myerror[80];
		scanf_s("%s",myerror); // wait for user's keyboard response

		return true; // ACD:  set to true instead of false, to "mark" the error condition; although currently the calling program doesn't use the returned value.
	}

	// obtain the binary
	size_t num_of_binaries=0; 
	clGetProgramInfo(cpProgram, CL_PROGRAM_NUM_DEVICES, sizeof(size_t), &num_of_binaries, NULL);

	size_t *binary_sizes = new size_t[num_of_binaries];

	ciErrNum = clGetProgramInfo(cpProgram, CL_PROGRAM_BINARY_SIZES, num_of_binaries*sizeof(size_t*), binary_sizes, NULL);
	if (ciErrNum!=CL_SUCCESS)
	{
		printf ("Error: clGetProgramInfo %s at line %d, file %s\n", GetErrorCode (ciErrNum), __LINE__, __FILE__); 
		return false;
	}
	
	char **binaries = new char*[num_of_binaries];
	for (size_t i=0; i<num_of_binaries; i++)
		binaries[i] = new char[binary_sizes[i]];

	ciErrNum = clGetProgramInfo(cpProgram, CL_PROGRAM_BINARIES, (size_t)num_of_binaries*sizeof(unsigned char*), binaries, NULL);
	if (ciErrNum!=CL_SUCCESS)
	{
		// write out standard error, Build Log and PTX, then cleanup and exit
		printf ("clGetProgramInfo at %d in %s\n", __LINE__, __FILE__);
		return false;
	}

	// prepare the output file name, .cl --> .co
	string strFileName(FileName);

	size_t found;
	found=strFileName.find_last_of(".cl");
	strFileName.replace(found-1, 2, "co");
	
	// write binaries to files
	ofstream fileStream(strFileName.c_str(), ofstream::binary);

	if(fileStream.is_open() == true)   
	{
		for (size_t i=0; i<num_of_binaries; i++)
		{
			fileStream.write ((const char *)(binary_sizes+i), (size_t)sizeof(binary_sizes[i]));
		}
		for (size_t i=0; i<num_of_binaries; i++)
			fileStream.write ((const char*)binaries[i], (size_t)binary_sizes[i]);

		fileStream.close();
	}
	else
	{
		printf ("Error: could not create binary file %s\n", strFileName.c_str());
		return false;
	}


	// cleanup
	delete [] binary_sizes;
	for (size_t i=0; i<num_of_binaries; i++)
		delete [] binaries[i];
	delete [] binaries;

	return true;
}

bool CL_MAGMA_RT::BuildFromBinaries(const char *FileName)
{
	if (FileName==NULL)
	{
		printf ("Error: file name empty on line %d in %s\n", __LINE__, __FILE__);
		return false;
	}
	
	cl_uint num_of_binaries = 0;
	size_t *binary_sizes;
	unsigned char **binaries;

	// load binary from file
	ifstream fileStream(FileName, ios::binary | ios::in | ios::ate);

	if (fileStream.is_open() == true)
	{
		fileStream.seekg(0, ios::beg);

		num_of_binaries = ciDeviceCount;

		binary_sizes = new size_t[num_of_binaries];
		for (size_t i = 0; i<num_of_binaries; i++)
			fileStream.read((char*)(binary_sizes + i), sizeof(binary_sizes[0]));

		binaries = new unsigned char*[num_of_binaries];
		for (size_t i = 0; i<num_of_binaries; i++)
		{
			binaries[i] = new unsigned char[binary_sizes[i]];
			fileStream.read((char*)binaries[i], (size_t)binary_sizes[i]);
		}

		fileStream.close();
	}
	else
	{
		printf("Error could not open %s\n", FileName);
		return false;
	}

	// build program from binaries
	cl_program cpProgram = clCreateProgramWithBinary(
		cxGPUContext, num_of_binaries, cdDevices,
		(const size_t*)binary_sizes, (const unsigned char **)binaries, &ciErrNum, &ciErrNum2);

	ciErrNum = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
	if (ciErrNum != CL_SUCCESS || ciErrNum2 != CL_SUCCESS)
	{
		// write out standard error, Build Log and PTX, then cleanup and exit

		return false;
	}

	
	// put program in the pool
	ProgramPool[string(FileName)] = cpProgram;

	delete [] binary_sizes;
	for (size_t i=0; i<num_of_binaries; i++)
		delete [] binaries[i];
	delete [] binaries;

	return true;
}

bool CL_MAGMA_RT::BuildFromSources(const char *FileName)
{
	if (FileName == NULL)
	{
		printf("Error: file name empty on line %d in %s\n", __LINE__, __FILE__);
		return false;
	}

	// read in the kernel source
	string fileStrings;

	fileStrings = fileToString(FileName);
	const char *filePointers = fileStrings.c_str();

	// Create the program
	cl_program cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char**)&filePointers, NULL, &ciErrNum);
	if (ciErrNum != CL_SUCCESS)
	{
		printf("Error: clCreateProgramWithSource at %d in %s\n", __LINE__, __FILE__);
		return false;
	}

	ciErrNum = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
	if (ciErrNum != CL_SUCCESS)
	{
		printf("clBuildProgram error at %d in %s : ", __LINE__, __FILE__);
		switch (ciErrNum) {
		case CL_INVALID_PROGRAM:
			printf("CL_INVALID_PROGRAM\n");
			break;

		case CL_INVALID_VALUE:
			printf("CL_INVALID_VALUE\n");
			break;

		case CL_INVALID_DEVICE:
			printf("CL_INVALID_DEVICE\n");
			break;

		case CL_INVALID_BINARY:
			printf("CL_INVALID_BINARY\n");
			break;

		case CL_INVALID_BUILD_OPTIONS:
			printf("CL_INVALID_BUILD_OPTIONS\n");
			break;

		case CL_INVALID_OPERATION:
			printf("CL_INVALID_OPERATION\n");
			break;

		case CL_COMPILER_NOT_AVAILABLE:
			printf("CL_COMPILER_NOT_AVAILABLE\n");
			break;

		case CL_BUILD_PROGRAM_FAILURE:
			printf("CL_BUILD_PROGRAM_FAILURE\n");
			break;

		case CL_OUT_OF_HOST_MEMORY:
			printf("CL_OUT_OF_HOST_MEMORY\n");
			break;

		default:
			printf(" Unknown error.\n");
			break;
		}

		printf("NOTE ERROR:  Enter any non-space text to continue.\n");
		char myerror[80];
		scanf_s("%s", myerror); // wait for user's keyboard response

		return true; // ACD:  set to true instead of false, to "mark" the error condition; although currently the calling program doesn't use the returned value.
	}

	// put program in the pool
	ProgramPool[string(FileName)] = cpProgram;

	return true;
}

/*
 * map kernel name to file
 * incomplete
 */
bool CL_MAGMA_RT::BuildKernelMap(const char *FileNameList)
{
	if (FileNameList==NULL)
		return false;

	/*
	//read from clfile for a list of cl files to compile  
	vector<string> FileNames;
	int NumOfFiles = GatherFilesToCompile (FileNameList, FileNames);

	if (NumOfFiles==0)
		return false;
		*/

	return true;
}

bool CL_MAGMA_RT::CreateKernel(const char *KernelName)
{
	if (!HasBeenInitialized)
	{
		printf ("Error: Uninitialized kernel\n");
		return false;
	}

	cl_program cpProgram = NULL;
	//printf ("getting kernel %s from %s\n", KernelName, Kernel2FileNamePool[string(KernelName)].c_str());
	cpProgram = ProgramPool[ Kernel2FileNamePool[string(KernelName)]];
	if (cpProgram==NULL)
	{
		printf ("Error: could not find program for kernel %s\n", KernelName);
		return false;
	}

	KernelPool[string(KernelName)] = clCreateKernel(cpProgram, KernelName, &ciErrNum);
	if (ciErrNum != CL_SUCCESS)
	{
		printf ("Error: could not create kernel %s\n", KernelName);
		return false;
	}

	return true;
}


//*****************************************************
//*****************************************************
//*****************************************************
//*****************************************************
//*****************************************************
//******************* ACD 2013-07-05  *****************
//************* IN SUPPORT OF _ACCESS() SUBR **********
//*****************************************************
//*****************************************************
//*****************************************************
//*****************************************************
//*****************************************************
//**********     double_backslashes()    ****
//*****************************************************
//*****************************************************
// function to display message box showing helpful info
string double_backslashes(string path){
	string newpath;  
	for ( unsigned int i = 0; i <= path.length() - 1 ; i++) 
	// -1 to skip the null terminating character at end of string
	{
		newpath.append(path,i,1);
		if(path.at(i) == '\\') // string.at(i) yields a char
		// or could use == 0x5C
		{
		   newpath.append(path,i,1);
//		   newpath.append(path.at(i));
		}
	}
	// 
	return newpath;
}
