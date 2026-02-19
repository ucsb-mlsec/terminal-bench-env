#ifndef INTERFACE_H
#define INTERFACE_H

#include <string>
#include <vector>

// Abstract interface for data processing
class IDataProcessor
{
public:
    virtual ~IDataProcessor() {}
    
        virtual void initialize(const std::string &configPath, int maxThreads, bool enableLogging) = 0;
    
  virtual bool processData(std::vector<int> *inputData,std::vector<int>* outputData,const std::string& processingMode)=0;
    
    virtual int calculateChecksum(const std::vector<unsigned char>& data, bool useAdvancedAlgorithm, int polynomialSeed) = 0;
    
            virtual void setParameter(const std::string &paramName,double value)=0;
    
    virtual double getParameter( const std::string& paramName ) const = 0;
    
  virtual bool validateConfiguration(const std::string &configFile,std::vector<std::string> *errors) const = 0;
    
        virtual void registerCallback(void(*callback)(int,const std::string &),void * userData)=0;
    
    virtual std::string getStatusReport(bool includeDetailedMetrics, bool includeTimestamps, int verbosityLevel) const = 0;
    
  virtual void applyTransformation(std::vector<double> * matrix,int rows,int cols,const std::string& transformationType)=0;
    
            virtual bool exportResults(const std::string &outputPath, const std::vector<int>& results, bool compressOutput) = 0;
    
    virtual void* getRawDataPointer( )const=0;
    
        virtual int performComplexOperation(const std::vector<std::string>& inputStrings, std::vector<int> *outputIntegers, bool enableCaching, int timeoutMilliseconds) = 0;
    
  virtual void cleanup( ) = 0;
};

#endif