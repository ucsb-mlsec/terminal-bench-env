#ifndef SIGNAL_LIB_H
#define SIGNAL_LIB_H

#ifdef _WIN32
    #ifdef SIGNAL_LIB_EXPORTS
        #define SIGNAL_API __declspec(dllexport)
    #else
        #define SIGNAL_API __declspec(dllimport)
    #endif
#else
    #define SIGNAL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

SIGNAL_API void processSignal(float* data, int size);

SIGNAL_API const char* getLibraryVersion();

#ifdef __cplusplus
}
#endif

#endif // SIGNAL_LIB_H