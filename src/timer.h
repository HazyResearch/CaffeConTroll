
#ifndef elly_Timer_h
#define elly_Timer_h

#include <time.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#ifdef __MACH__
#include <sys/time.h>
//clock_gettime is not implemented on OSX
int clock_gettime(int /*clk_id*/, struct timespec* t);

#define CLOCK_MONOTONIC 0
#endif

/**
 * Timer class that keeps track of time
 */
class Timer {
  protected:
    struct timespec _start;
    struct timespec _end;

  public:
    Timer();

    /**
     * Restart the timer
     */
    void restart();

    /**
     * Returns time elapsed (in seconds)
     */
    float elapsed();
};

#endif
