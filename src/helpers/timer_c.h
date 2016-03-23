/*
* Timer class based on ctime
* showed to be working on all platforms
* although in current implementation far from perfect
*/
#pragma once
#ifndef TIMER_C_H_
#define TIMER_C_H_
#include <ctime>
namespace timer
{
	struct Timer{
	protected:
		std::clock_t start_t, end_t;
		double duration;
		std::string name;
	public:
		void start();
		void stop();
		void setName(std::string s);
		float elapsed();
		void print(const std::string &name = "");
	};

	void Timer::start(){
		this->start_t = std::clock();
	}
	void Timer::stop(){
		this->end_t = std::clock();
	}
	void Timer::setName(std::string s){
		this->name = s;
	}
	float Timer::elapsed(){
		return this->duration = (this->end_t - this->start_t) / (double)CLOCKS_PER_SEC;
		
	}
	void Timer::print(const std::string &name){
		this->duration = (this->end_t - this->start_t) / (double)CLOCKS_PER_SEC;
		if (name == "")
			std::cout << this->name << ": " << this->duration << " ms" << std::endl;
		else
			std::cout << name << ": " << this->duration << " ms" << std::endl;
	}

	std::vector<Timer> timers;
	Timer t;
	void start(std::string s){
		t.setName(s);
		timers.push_back(t);
		timers[timers.size()-1].start();
	}

	void stop(std::string s){
		timers[timers.size() - 1].stop();
	}

	void printToScreen(){
		for (int i = 0; i < timers.size(); i++){
			timers[i].print();
		}
	}

	void reset(){
		timers.clear();
	}

	float elapsed(){
		return timers[timers.size() - 1].elapsed();
	};
}
#endif // TIMER_C_H_