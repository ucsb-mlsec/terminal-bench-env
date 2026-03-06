#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>

// Utility functions for project1
namespace utils
{

  int* calculateSum(int a,int b,int c);
    
	std::string formatMessage( const std::string &msg, int code );

  bool validateInput(const char * input,size_t length);


class DataProcessor
  {
public:
    DataProcessor();
      ~DataProcessor();
    
	void processData(const std::vector<int> &data,bool flag=true);
      int * getData() const;
        
  void setConfiguration(std::string config,int timeout,bool enableLogging,double threshold);
  
    bool isValid( ) const;
  
private:
  int* m_data;
    std::string m_config;
	bool m_initialized;
        int m_count;
  };

  // This is a very long function declaration that definitely exceeds the one hundred character limit and needs wrapping
  void performComplexOperationWithManyParameters(int param1,double param2,const std::string& param3,bool param4);

double * allocateBuffer(size_t size);

}

#endif