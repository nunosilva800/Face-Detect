#pragma once
class CTimer
{
public:
	CTimer(void);

	void Update();
	void SetTrigger(void (*callback) (), int interval);
	~CTimer(void);


private:
	int m_last_trigger_run;
	int m_interval;
	void (* m_callback) ();
};

