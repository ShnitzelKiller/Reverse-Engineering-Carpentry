#include <chrono>

#define DECLARE_TIMING(s)  std::chrono::time_point<std::chrono::high_resolution_clock> timeStart_##s; double timeDiff_##s
//; double timeTally_##s = 0; int countTally_##s = 0
#define START_TIMING(s)    timeStart_##s = std::chrono::high_resolution_clock::now()
#define STOP_TIMING(s)     timeDiff_##s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - timeStart_##s).count()
//; timeTally_##s += timeDiff_##s; countTally_##s++
#define GET_TIMING(s)      timeDiff_##s
#define PRINT_TIMING(s)    std::cout << "finished in " << timeDiff_##s << " seconds" << std::endl
//#define GET_AVERAGE_TIMING(s)   (double)(countTally_##s ? timeTally_##s/ ((double)countTally_##s * cvGetTickFrequency()*1000.0) : 0)
//#define CLEAR_AVERAGE_TIMING(s) timeTally_##s = 0; countTally_##s = 0
