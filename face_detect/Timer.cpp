#include "Timer.h"


#include <time.h>

CTimer::CTimer(void) : m_callback(0), m_last_trigger_run(0)
{
}


CTimer::~CTimer(void)
{
}


void CTimer::Update()
{
	time_t timer;

	time(&timer);

	if((timer - m_last_trigger_run) > m_interval)
	{
		if(m_callback)
			m_callback();
		
		m_last_trigger_run = timer;
	}
}

void CTimer::SetTrigger(void (*callback ) (), int interval)
{
	m_callback = callback;
	m_interval = interval;

	time_t timer;
	time(&timer);
	m_last_trigger_run = timer;
}
