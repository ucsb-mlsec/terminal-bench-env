#ifndef   TYPES_H
#define TYPES_H

#include <string>
#include <vector>
#include <memory>

typedef unsigned long long int UnsignedLongLongIntegerTypeForExtremelyLongDataProcessingOperations;
typedef std::vector<std::string> StringVector;

using StringPtr=std::shared_ptr<std::string>;
using   IntVector   =   std::vector<int>;

struct DataPoint
{
int x;
    int y;
        int z;
  std::string label;
};

class   ConfigurationManager   
  {
public:
ConfigurationManager();
    ~ConfigurationManager( );
        void   loadConfiguration( const std::string   &filename );
  bool validateSettings( );
private:
std::vector<std::string>settings;
int  * configPointer;
        std::string&  getReference( );
};

struct VeryLongStructureNameThatDefinitelyExceedsTheMaximumColumnLimitAndNeedsToBeReformatted{
unsigned int first_field;int second_field;std::string third_field;double fourth_field;};

typedef struct   {
    int identifier;
std::string   name;
      double   value;
  bool    isActive;
} RecordType;

class   TemplateClass  {
  public:
      TemplateClass(int*ptr,std::string&ref):pointer(ptr),reference(ref){}
int*pointer;std::string&reference;
};

#endif