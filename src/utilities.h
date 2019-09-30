#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <stdexcept>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

#define InvPi			  0.31830988618379067154;
#define Inv2Pi			  0.15915494309189533577;
#define Inv4Pi			  0.07957747154594766788;

#define DIRECT_LIGHT 0
#define TRI_2D_Array 0

typedef glm::vec3 Color3f;
typedef glm::vec3 Point3f;
typedef glm::vec3 Normal3f;

namespace utilityCore {
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
	//float AbsDot(const glm::vec3& a, const glm::vec3& b);

	 //performance timer from hw2
	class PerformanceTimer
	{
	public:
		PerformanceTimer() {
			cudaEventCreate(&event_start);
			cudaEventCreate(&event_end);
		}

		~PerformanceTimer()
		{
			cudaEventDestroy(event_start);
			cudaEventDestroy(event_end);
		}

		void startCpuTimer()
		{
			if (cpu_timer_started) { throw std::runtime_error("CPU timer already started"); }
			cpu_timer_started = true;

			time_start_cpu = std::chrono::high_resolution_clock::now();
		}

		void endCpuTimer() {
			time_end_cpu = std::chrono::high_resolution_clock::now();

			if (!cpu_timer_started) { throw std::runtime_error("CPU timer not started"); }

			std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
			prev_elapsed_time_cpu_milliseconds =
				static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

			cpu_timer_started = false;
		}

		void startGpuTimer()
		{
			if (gpu_timer_started) { throw std::runtime_error("GPU timer already started"); }
			gpu_timer_started = true;

			cudaEventRecord(event_start);
		}

		void endGpuTimer()
		{
			cudaEventRecord(event_end);
			cudaEventSynchronize(event_end);

			if (!gpu_timer_started) { throw std::runtime_error("GPU timer not started"); }

			cudaEventElapsedTime(&prev_elapsed_time_gpu_milliseconds, event_start, event_end);
			gpu_timer_started = false;
		}

		float getCpuElapsedTimeForPreviousOperation() //noexcept //(damn I need VS 2015
		{
			return prev_elapsed_time_cpu_milliseconds;
		}

		float getGpuElapsedTimeForPreviousOperation() //noexcept
		{
			return prev_elapsed_time_gpu_milliseconds;
		}

		// remove copy and move functions
		PerformanceTimer(const PerformanceTimer&) = delete;
		PerformanceTimer(PerformanceTimer&&) = delete;
		PerformanceTimer& operator=(const PerformanceTimer&) = delete;
		PerformanceTimer& operator=(PerformanceTimer&&) = delete;

	private:
		cudaEvent_t event_start = nullptr;
		cudaEvent_t event_end = nullptr;

		using time_point_t = std::chrono::high_resolution_clock::time_point;
		time_point_t time_start_cpu;
		time_point_t time_end_cpu;

		bool cpu_timer_started = false;
		bool gpu_timer_started = false;

		float prev_elapsed_time_cpu_milliseconds = 0.f;
		float prev_elapsed_time_gpu_milliseconds = 0.f;
	};

	template<typename T>
	void printElapsedTime(T time, std::string note = "")
	{
		std::cout << "   elapsed time: " << time << "ms    " << note << std::endl;
	}
}
