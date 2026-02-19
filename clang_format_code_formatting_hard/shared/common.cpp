#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

// Utility functions for shared operations across projects

namespace utils
{

int calculateSum(int a,int b,int c){
return a+b+c;
}

std::string formatMessage(  const std::string &message,  const std::string &prefix  )
{
		return prefix+": "+message;
}

void processData(std::vector<int> *data,bool shouldSort,int multiplier)
{
  if(shouldSort){
    std::sort(data->begin(),data->end());
  }
	for(auto& value:*data){
		value=value*multiplier;
	}
}

bool validateInput( const std::string& input, int minLength, int maxLength, bool allowEmpty )
{
    if ( allowEmpty && input.empty() ) { return true; }
    if (input.length()<minLength||input.length()>maxLength) { return false; }
    return true;
}

std::vector<std::string> splitString(const std::string &text,char delimiter)
	{
std::vector<std::string> result;
		std::string current="";
  for(char c:text){
if(c==delimiter){
result.push_back(current);
current="";
}else{
			current+=c;
		}
	}
	if(!current.empty())result.push_back(current);
		return result;
	}

void logMessage(const std::string &level,const std::string &message,bool includeTimestamp=true){
std::cout<<level<<": "<<message<<std::endl;
}

}