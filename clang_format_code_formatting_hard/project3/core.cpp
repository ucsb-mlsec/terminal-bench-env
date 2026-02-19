#include <iostream>
#include <vector>
#include <string>
#include <memory>

// This is a poorly formatted C++ file with inconsistent styles everywhere
namespace project3
{

class DataProcessor {
  public:
    DataProcessor(int*data, size_t size) : m_data(data), m_size(size) {}
    
    void process();
    int * getData() { return m_data; }
    
  private:
    int* m_data;
    size_t m_size;
};

void DataProcessor::process()
{
  for(size_t i=0;i<m_size;++i)
    {
      if(m_data[i]>100){
        m_data[i]=100;
      }
    }
}

// Function with terrible formatting and super long line that definitely exceeds the 100 character limit we need to enforce
int * performComplexCalculation(int*input, int count, double multiplier, bool shouldNormalize, std::string debugMessage)
{
	int*result=new int[count];
  for (int i = 0; i < count; ++i) 
  {
    if( shouldNormalize )
      {
      result[ i ] = static_cast<int>(input[i]*multiplier);
    }
        else{
          result[i]=input[i];
	}
  }
  return result;
}

void processDataWithComplexLogic(std::vector<int>&data, int threshold)
{
    for(auto& value : data){
        if( value < threshold ) {
      value *= 2;
        }
      else if(value>threshold*2){
            value/=2;
    } else {
            value = threshold;
        }
    }
}

}  // namespace project3

int main(int argc,char**argv) {
  std::cout<<"Starting data processing application with multiple formatting issues"<<std::endl;
  
    int data[]={10,20,30,40,50,60,70,80,90,100,110,120};
  project3::DataProcessor processor(data,12);
    processor.process();
    
    std::vector<int>vec={1,2,3,4,5};
  project3::processDataWithComplexLogic( vec, 3 );
  
  return 0;
}