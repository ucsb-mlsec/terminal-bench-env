#include <iostream>
#include <vector>
#include <string>
#include   <algorithm>

using namespace std;

// Helper function with inconsistent brace placement
int calculateSum(vector<int>& numbers)
{
  int total = 0;
    for(int i=0;i<numbers.size();i++){
      total+=numbers[i];
    }
  return total;
}

// Function with long line and inconsistent spacing
double calculateAverage(vector<int> &numbers) {
    if(numbers.empty()) return 0.0;
    int sum=calculateSum(numbers);
    return static_cast<double>(sum)/static_cast<double>(numbers.size());  // This is a very long comment that exceeds the typical column limit of 100 characters
}

// Inconsistent pointer declaration style
string* createString(const char *input) {
  string* result = new string(input);
    return result;
}



void processData(int* data,int size)
{
    for (int i = 0; i < size; ++i) 
    {
      if(data[i]>10){
            data[i] = data[i] * 2;
      }
      else if (data[i] < 0) 
      {
        data[i]=0;
      }
        else{
          data[i]++;
        }
    }
}

// Main function with mixed indentation
int main(int argc, char** argv) {
  vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    cout<<"Starting program..."<<endl;
  
  int sum = calculateSum(numbers);
    double avg=calculateAverage(numbers);
    
        cout << "Sum: " << sum << endl;
  cout<<"Average: "<<avg<<endl;
  
    int* dataArray = new int[5];
  for(int i=0;i<5;i++)
    {
      dataArray[i]=i*3-5;
    }
    
  processData(dataArray, 5);
  
    cout << "Processed data: ";
  for (int i = 0; i < 5; ++i) {
        cout << dataArray[i] << " ";
  }
    cout<<endl;
    
  string *message=createString("Hello, World!");
    cout << *message << endl;
  
    delete message;
  delete[] dataArray;
  
    return 0;
}