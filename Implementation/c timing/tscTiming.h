#ifndef tscTiming_H
#define tscTiming_H


#include <stdint.h>

#if !defined(__i386__) && !defined(__x86_64__)
#error tscTiming.h only supports x86 and x86-64
#endif


__attribute__((gnu_inline, always_inline));
static uint64_t __inline__ tscTiming_start(void)
{
	uint32_t tl, th;
	__asm__ volatile ("cpuid;"
			"rdtsc;"
			"movl %%edx, %[th];"
			"movl %%eax, %[tl];"
			: /* outputs */
			[th] "=r" (th),
			[tl] "=r" (tl)
			: /* inputs */
			: /* clobbers */
			"rax", "rbx", "rcx", "rdx"
			);
	return (((uint64_t)(th)) << 32) | tl;
}


__attribute__((gnu_inline, always_inline))
static uint64_t __inline__ tsc_Timing_stop(void)
{
	uint32_t tl, th;
	__asm__ volatile ("rdtscp;"
			"movl %%edx, %[th];"
			"movl %%eax, %[tl];"
			"cpuid;"
			: /* outputs */
			[th] "=r" (th),
			[tl] "=r" (tl)
			: /* inputs */
			: /* clobbers */
			"rax", "rbx", "rcx", "rdx"
			);
	return (((uint64_t)th) << 32) | tl;
}

#endif