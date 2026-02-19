#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <vector>
#include <memory>

// Parser class for handling input data
class Parser {
public:
    Parser();
    Parser(const std::string& filename);
    ~Parser();
    
    bool parse(const std::string& input);
    std::string getLastError() const;
    void reset();
    
    // Get parsed tokens
    std::vector<std::string> getTokens()const;
    
    int getTokenCount( ) const;
    
  bool isValid() const;
  
    void setDebugMode(bool enabled);
    
private:
  // Helper methods
  bool validateInput(const std::string& input);
    void tokenize(const std::string& input);
  bool processToken(const std::string& token);
    
    void clearErrors();
  
    /* Internal state management */
    void updateState();
    
  // Member variables
  std::string m_filename;
    std::string m_lastError;
    std::vector<std::string> m_tokens;
  int m_tokenCount;
    bool m_isValid;
  bool m_debugMode;
    
    // Parser state
  enum State{
        STATE_IDLE,
    STATE_PARSING,
        STATE_ERROR
    };
    
  State m_currentState;
    std::unique_ptr<int> m_lineNumber;
};

#endif // PARSER_H