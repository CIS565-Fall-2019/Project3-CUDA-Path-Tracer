/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"

#include <iostream>
#include <fstream>
using namespace std;

//const int SIZE = 1 << 20; // feel free to change the size of array
//const int NPOT = SIZE - 3; // Non-Power-Of-Two
//int *a = new int[SIZE];
//int *b = new int[SIZE];
//int *c = new int[SIZE];

int SIZE ; 
int NPOT ;
int *a ;
int *b ;
int *c ;


int main(int argc, char* argv[]) {
	// Scan tests

	printf("\n");
	printf("****************\n");
	printf("** SCAN TESTS **\n");
	printf("****************\n");

	//ofstream outputFile1("Naive_Scan.txt");
	//ofstream outputFile11("Naive_Scan_NP.txt");
	//ofstream outputFile2("WorkEff_Scan.txt");
	//ofstream outputFile22("WorkEff_Scan_NP.txt");


	for (int sz = 20; sz < 21; sz++) {

		//int SIZE = 1 << 20; // feel free to change the size of array
		//SIZE = 1 << sz;

		SIZE = 1<<25;


		NPOT = SIZE - 3; // Non-Power-Of-Two
		a = new int[SIZE];
		b = new int[SIZE];
		c = new int[SIZE];

		genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
		a[SIZE - 1] = 0;
		printArray(SIZE, a, true);
		
		// CPU Scans ==================================================================================================

		// initialize b using StreamCompaction::CPU::scan you implement
		// We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
		// At first all cases passed because b && c are all zeroes.

		zeroArray(SIZE, b);
		printDesc("cpu scan, power-of-two");
		StreamCompaction::CPU::scan(SIZE, b, a);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
		//outputFile1 << sz << " | size " << SIZE << " | " << StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation() << " " << "(std::chrono Measured)" << endl;
		printArray(SIZE, b, true);

		zeroArray(SIZE, c);
		printDesc("cpu scan, non-power-of-two");
		StreamCompaction::CPU::scan(NPOT, c, a);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
		//outputFile2 << sz << " | size " << SIZE << " | " << StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation() << " " << "(std::chrono Measured)" << endl;
		printArray(NPOT, b, true);
		printCmpResult(NPOT, b, c);


		// GPU naive Scan ===========================================================================================
		
		zeroArray(SIZE, c);
		printDesc("GPU naive scan, power-of-two");
		StreamCompaction::Naive::scan(SIZE, c, a);
		printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//outputFile1 << sz << " | size " << SIZE << "  StreamCompaction::Naive::scan Poweof2  " << SIZE << " | " << StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation() << " (CUDA Measured)" << endl;
		//printArray(SIZE, c, true);
		printCmpResult(SIZE, b, c);

		// For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
		//onesArray(SIZE, c);
		//printDesc("1s array for finding bugs");
		//StreamCompaction::Naive::scan(SIZE, c, a);
		//printArray(SIZE, c, true); 

		zeroArray(SIZE, c);
		printDesc("GPU naive scan, non-power-of-two");
		StreamCompaction::Naive::scan(NPOT, c, a);
		printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//outputFile11 << sz << " | size " << SIZE << "  StreamCompaction::Naive::scan NonPoweof2  " << SIZE << " | " << StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation() << " (CUDA Measured)" << endl;
		//printArray(SIZE, c, true);
		printCmpResult(NPOT, b, c);

		// GPU Work Eff Scan ===========================================================================================

		zeroArray(SIZE, c);
		printDesc("work-efficient scan, power-of-two");
		StreamCompaction::Efficient::scan(SIZE, c, a);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//outputFile2 << sz << " | size " << SIZE <<"  StreamCompaction::Efficient::scan Poweof2  " << SIZE << " | " << StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() << " (CUDA Measured)" << endl;
		printCmpResult(SIZE, b, c);

		zeroArray(SIZE, c);
		printDesc("work-efficient scan, non-power-of-two");
		StreamCompaction::Efficient::scan(NPOT, c, a);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//outputFile22<< sz << " | size " << SIZE << "  StreamCompaction::Efficient::scan NonPoweof2  " << SIZE << " | " << StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() << " (CUDA Measured)" << endl;
		//printArray(NPOT, c, true);
		printCmpResult(NPOT, b, c);


		// GPU Thrust Scan ===========================================================================================

		zeroArray(SIZE, c);
		printDesc("thrust scan, power-of-two");
		StreamCompaction::Thrust::scan(SIZE, c, a);
		printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//outputFile1 << sz << " | size " << SIZE << " | " << StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation() << " " << "(CUDA Measured)" << endl;
		//printArray(SIZE, c, true);
		printCmpResult(SIZE, b, c);

		zeroArray(SIZE, c);
		printDesc("thrust scan, non-power-of-two");
		StreamCompaction::Thrust::scan(NPOT, c, a);
		printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//outputFile2 << sz << " | size " << SIZE << " | " << StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation() << " " << "(CUDA Measured)" << endl;
		//printArray(NPOT, c, true);
		printCmpResult(NPOT, b, c);

	}

    printf("\n");
    printf("*****************************\n");

    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

	//ofstream outputFile1("SC_CPU.txt");
	//ofstream outputFile11("SC_CPU_NP.txt");
	//ofstream outputFile2("SC_CPU_withScan.txt");
	//ofstream outputFile3("SC_WorkEff.txt");
	//ofstream outputFile33("SC_WorkEff_NP.txt");
	
	for (int sz = 20; sz < 21; sz++) {

		//int SIZE = 1 << 20; // feel free to change the size of array
		SIZE = 1 << 25;

		NPOT = SIZE - 3; // Non-Power-Of-Two
		a = new int[SIZE];
		b = new int[SIZE];
		c = new int[SIZE];

		// Compaction tests

		genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
		a[SIZE - 1] = 0;
		printArray(SIZE, a, true);

		int count, expectedCount, expectedNPOT;

		// initialize b using StreamCompaction::CPU::compactWithoutScan you implement
		// We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
		zeroArray(SIZE, b);
		printDesc("cpu compact without scan, power-of-two");
		count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
		//outputFile1 << sz << " " << SIZE << " " << StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation() << " " << "(std::chrono Measured)" << endl;
		expectedCount = count;
		printArray(count, b, true);
		printCmpLenResult(count, expectedCount, b, b);

		zeroArray(SIZE, c);
		printDesc("cpu compact without scan, non-power-of-two");
		count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
		//outputFile11 << sz << " " << SIZE << " " << StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation() << " " << "(std::chrono Measured)" << endl;
		expectedNPOT = count;
		printArray(count, c, true);
		printCmpLenResult(count, expectedNPOT, b, c);

		zeroArray(SIZE, c);
		printDesc("cpu compact with scan");
		count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
		//outputFile2 << sz << " " << SIZE << " " << StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation() << " " << "(std::chrono Measured)" << endl;
		printArray(count, c, true);
		printCmpLenResult(count, expectedCount, b, c);

		zeroArray(SIZE, c);
		printDesc("work-efficient compact, power-of-two");
		count = StreamCompaction::Efficient::compact(SIZE, c, a);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//outputFile3 << sz << " " << SIZE << " " << StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() << " " << "(CUDA Measured)" << endl;
		//printArray(count, c, true);
		printCmpLenResult(count, expectedCount, b, c);

		zeroArray(SIZE, c);
		printDesc("work-efficient compact, non-power-of-two");
		count = StreamCompaction::Efficient::compact(NPOT, c, a);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//outputFile33 << sz << " " << SIZE << " " << StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() << " " << "(CUDA Measured)" << endl;
		//printArray(count, c, true);
		printCmpLenResult(count, expectedNPOT, b, c);
	}
	
    system("pause"); // stop Win32 console from closing on exit
	delete[] a;
	delete[] b;
	delete[] c;
}
