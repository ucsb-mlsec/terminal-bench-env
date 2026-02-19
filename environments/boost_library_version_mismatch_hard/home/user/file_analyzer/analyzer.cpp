#include <boost/filesystem.hpp>
#include <iostream>
#include <iomanip>

using namespace boost::filesystem;
using namespace std;

int main()
{
    cout << "File Analyzer - Directory Analysis Tool" << endl;
    cout << "=======================================" << endl;
    
    path current_dir(".");
    
    try {
        if (!exists(current_dir) || !is_directory(current_dir)) {
            cerr << "Error: Current directory not accessible" << endl;
            return 1;
        }
        
        cout << "\nAnalyzing directory: " << current_dir.directory_string() << endl;
        cout << "\nFiles found:" << endl;
        cout << "----------------------------------------" << endl;
        
        unsigned long total_size = 0;
        int file_count = 0;
        
        directory_iterator end_iter;
        for (directory_iterator dir_itr(current_dir); dir_itr != end_iter; ++dir_itr) {
            try {
                if (is_regular_file(dir_itr->status())) {
                    path file_path = dir_itr->path();
                    unsigned long size = file_size(file_path);
                    
                    // Using old Boost v2 API - filename() returns path object
                    // that can be streamed directly in v2
                    cout << setw(30) << left << file_path.filename() 
                         << " : " << setw(12) << right << size << " bytes" << endl;
                    
                    total_size += size;
                    file_count++;
                }
                else if (is_directory(dir_itr->status())) {
                    // Show directories with native path string format
                    cout << "[DIR] " << dir_itr->path().native_file_string() << endl;
                }
            }
            catch (const filesystem_error& ex) {
                cerr << "Error accessing: " << dir_itr->path().file_string() << endl;
                cerr << ex.what() << endl;
            }
        }
        
        cout << "----------------------------------------" << endl;
        cout << "Total files: " << file_count << endl;
        cout << "Total size: " << total_size << " bytes";
        
        if (total_size > 1024*1024) {
            cout << " (" << fixed << setprecision(2) 
                 << (total_size / (1024.0 * 1024.0)) << " MB)";
        }
        else if (total_size > 1024) {
            cout << " (" << fixed << setprecision(2) 
                 << (total_size / 1024.0) << " KB)";
        }
        cout << endl;
        
    }
    catch (const filesystem_error& ex) {
        cerr << "Filesystem error: " << ex.what() << endl;
        return 1;
    }
    catch (const exception& ex) {
        cerr << "Error: " << ex.what() << endl;
        return 1;
    }
    
    return 0;
}