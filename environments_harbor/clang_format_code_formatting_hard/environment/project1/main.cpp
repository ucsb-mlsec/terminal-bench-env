#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

class DataProcessor
{
  public:
    DataProcessor(int capacity) : capacity_(capacity), count_(0) {}
    
  void addItem(const std::string &item)
  {
      if(count_ < capacity_){
        items_.push_back(item);
          count_++;
      }
  }

    int getCount( ) const { return count_; }
    
    void processItems()
    {
      for(int i=0;i<items_.size();++i)
      {
          std::cout<<"Processing item: "<<items_[i]<<std::endl;
      }
    }

  private:
	int capacity_;
  int count_;
    std::vector<std::string> items_;
};

int* createArray(int size)
{
  int *arr = new int[size];
    for (int i = 0; i < size; i++)
    {
      arr[i] = i * 2;
    }
  return arr;
}

int main( )
{
    DataProcessor processor(10);
    
  processor.addItem("first");
    processor.addItem("second");
  processor.addItem("third");

    if (processor.getCount()>0)
    {
        std::cout << "Total items: " << processor.getCount() << std::endl;
        processor.processItems();
    }

    int *numbers = createArray(5);
    for(int i=0; i<5; ++i){
        std::cout << "Number at index " << i << " is " << numbers[i] << " which was calculated using a simple multiplication formula" << std::endl;
    }
    
    delete[] numbers;
  
  return 0;
}