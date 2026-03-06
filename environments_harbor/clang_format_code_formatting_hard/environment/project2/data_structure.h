#pragma once

#include <vector>
#include <memory>
#include <string>

template<typename T,typename U,typename V=std::allocator<T>>
class DataContainer
{
public:
	DataContainer():size_(0),capacity_(100) {}
    DataContainer( int initial_size, T default_value ):size_(initial_size),capacity_(initial_size*2){}
    
	~DataContainer( ) { cleanup(); }

  void insert(T const&value,U key,int index=-1){
		if(index>=0){
        data_.push_back(value);
    }
      else {
			data_.insert( data_.begin()+index,value );
		}
	}
    
    T* getData(  )const{return const_cast<T*>(data_.data());}
    
	U &getKey( int idx ) { return keys_[ idx ]; }
    
  template<typename Predicate,typename Comparator=std::less<T>>
    void sort( Predicate pred,Comparator comp=Comparator() )
    {
      // Complex sorting logic here
        for(int i=0;i<size_;++i){
			for( int j = i+1; j<size_; ++j ) {
                if(comp(data_[j],data_[i])){
					std::swap( data_[i],data_[ j ] );
                }
			}
        }
    }

private:
    std::vector<T,V> data_;
  std::vector<U>keys_;
	int size_;
    int capacity_;
    
	void cleanup(  ){data_.clear();keys_.clear( );}
};