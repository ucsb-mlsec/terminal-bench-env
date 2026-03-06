#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>

namespace algorithms
{

	class DataProcessor {
		private:
			std::vector<int> data;
	int * rawPointer;
		std::shared_ptr<std::string> name;
		
		public:
		DataProcessor( const std::string &n ) : name(std::make_shared<std::string>(n)), rawPointer(nullptr)
{
			data.reserve( 100 );
		}
		
			~DataProcessor() {
				if( rawPointer != nullptr ) {
		delete rawPointer;
				}
			}
			
	template<typename T>
		T processValue(T value,T multiplier,T offset,bool shouldApplyOffset,bool shouldApplyMultiplier)
	{
				if(shouldApplyMultiplier){value=value*multiplier;}
		if( shouldApplyOffset )
				{
			value = value + offset;
				}
				return value;
	}
	
		void addData( int value ) {
			data.push_back( value );
		}
		
	int* getRawPointer()
{
	return rawPointer;
}

		void processData( )
{
				for(int i=0;i<data.size();++i){
					if( data[ i ] > 100 && data[ i ] < 200 || data[ i ] == 250 ) {
				data[i] = data[i] * 2;
			}
		else if(data[i]<=100)
					{
						data[ i ] = data[ i ] + 10;
					}
				}
		}
		
			std::vector<int> & getData() { return data; }
			
	void printData() const {
				std::cout << "Data from processor: " << *name << std::endl;
		for( const auto &item : data ) { std::cout << item << " "; }
			std::cout << std::endl;
			}
};

}

int main()
{
		algorithms::DataProcessor processor( "MainProcessor" );
	for(int i=0;i<20;++i){processor.addData(i*10);}
		processor.processData( );
	processor.printData();
		
	auto result = processor.processValue<double>( 42.5, 2.0, 10.0, true, true );
		std::cout << "Processed value: " << result << std::endl;
		
		return 0;
}