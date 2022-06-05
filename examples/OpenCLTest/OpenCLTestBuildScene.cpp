/**************************************************************
** OpenCLTestBuildScene.cpp
**
** Started from base ExampleXGL code.
**
** Added OpenCL sample code from:
**   https://gist.github.com/ddemidov/2925717
**************************************************************/
#include "ExampleXGL.h"

#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

// Compute c = a + b.
static const char source[] =
"#if defined(cl_khr_fp64)\n"
"#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
"#elif defined(cl_amd_fp64)\n"
"#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
"#else\n"
"#  error double precision is not supported\n"
"#endif\n"
"kernel void add(\n"
"       ulong n,\n"
"       global const double *a,\n"
"       global const double *b,\n"
"       global double *c\n"
"       )\n"
"{\n"
"    size_t i = get_global_id(0);\n"
"    if (i < n) {\n"
"       c[i] = a[i] + b[i];\n"
"    }\n"
"}\n";

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	const size_t N = 1 << 20;

	try {
		std::vector<cl::Platform> platform;
		cl::Platform::get(&platform);

		if (platform.empty()) {
			xprintf("Doh! no OpenCL platform!\n");
		}

		// Get first available GPU device which supports double precision.
		cl::Context context;
		std::vector<cl::Device> devices;

		for (auto p : platform) {
			std::vector<cl::Device> d;

			try {
				p.getDevices(CL_DEVICE_TYPE_GPU, &d);

				for (auto d : d) {
					if (!d.getInfo<CL_DEVICE_AVAILABLE>()) 
						continue;

					std::string ext = d.getInfo<CL_DEVICE_EXTENSIONS>();

					if (
						ext.find("cl_khr_fp64") == std::string::npos &&
						ext.find("cl_amd_fp64") == std::string::npos
					) continue;

					devices.push_back(d);
					context = cl::Context(d);
				}
			}
			catch (...) {
				d.clear();
			}
		}


		if (devices.empty()) {
			std::cerr << "GPUs with double precision not found." << std::endl;
			return;
		}

		std::cout << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;

		// Create command queue.
		cl::CommandQueue queue(context, devices[0]);

		// Compile OpenCL program for found device.
		cl::Program program(context, cl::Program::Sources(
			1, std::make_pair(source, strlen(source))
		));


		try {
			program.build(devices);
		}
		catch (const cl::Error&) {
			std::cerr
				<< "OpenCL compilation error" << std::endl
				<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
				<< std::endl;
			return;
		}

		cl::Kernel add(program, "add");

		// Prepare input data.
		std::vector<double> a(N, 1);
		std::vector<double> b(N, 2);
		std::vector<double> c(N);

		// Allocate device buffers and transfer input data to device.
		cl::Buffer A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			a.size() * sizeof(double), a.data());

		cl::Buffer B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			b.size() * sizeof(double), b.data());

		cl::Buffer C(context, CL_MEM_READ_WRITE,
			c.size() * sizeof(double));

		// Set kernel parameters.
		add.setArg(0, static_cast<cl_ulong>(N));
		add.setArg(1, A);
		add.setArg(2, B);
		add.setArg(3, C);

		// Launch kernel on the compute device.
		queue.enqueueNDRangeKernel(add, cl::NullRange, N, cl::NullRange);

		// Get result back to host.
		queue.enqueueReadBuffer(C, CL_TRUE, 0, c.size() * sizeof(double), c.data());

		// Should get '3' here.
		std::cout << c[42] << std::endl;
	}
	catch (const cl::Error &err) {
		std::cerr
			<< "OpenCL error: "
			<< err.what() << "(" << err.err() << ")"
			<< std::endl;
		return;
}

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });
}
