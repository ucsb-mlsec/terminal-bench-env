#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

class Parser {
private:
    std::string input;
    size_t position;
    std::vector<std::string> tokens;
    
public:
    Parser(const std::string& data) : input(data), position(0) {}
    
    bool parse() {
        if(input.empty())
        {
            return false;
        }
        
        tokenize();
        return validate();
    }
    
    void tokenize()
    {
        std::string current = "";
        for(size_t i=0; i<input.length(); ++i)
        {
            char c = input[i];
            if(c == ' ' || c == '\t' || c == '\n') {
                if(!current.empty()) {
                    tokens.push_back(current);
                    current = "";
                }
            }
            else if(c == ',' || c == ';' || c == ':') {
                if(!current.empty()) { tokens.push_back(current); current = ""; }
                tokens.push_back(std::string(1, c));
            } else {
                current += c;
            }
        }
        if(!current.empty()) { tokens.push_back(current); }
    }
    
    bool validate()
    {
        int braceCount=0;
        int parenCount = 0;
        
        for(const auto& token : tokens) {
            if(token == "{") {
                braceCount++;
            }
            else if(token == "}") 
            {
                braceCount--;
                if(braceCount < 0) return false;
            }
            else if(token == "(") { parenCount++; }
            else if(token == ")") { parenCount--; if(parenCount<0) return false; }
        }
        
        return (braceCount == 0 && parenCount == 0);
    }
    
    std::vector<std::string>& getTokens() { return tokens; }
    
    std::string* findToken(const std::string& target) {
        for(auto& token : tokens)
        {
            if(token == target) return &token;
        }
        return nullptr;
    }
    
    void processCommands() {
        for(size_t idx = 0; idx < tokens.size(); ++idx) {
            std::string cmd = tokens[idx];
            
            switch(cmd[0]) {
                case 'S':
                case 's':
                    handleStart(idx);
                    break;
                case 'E':
                case 'e': { handleEnd(idx); break; }
                case 'P':
                case 'p':
                {
                    handleProcess(idx);
                    break;
                }
                default:
                    std::cerr << "Unknown command: " << cmd << std::endl;
            }
        }
    }
    
    void handleStart(size_t& index) {
        std::cout<<"Starting process at index "<<index<<std::endl;
        if(index + 1 < tokens.size() && tokens[index+1] == ":") { index += 2; }
    }
    
    void handleEnd(size_t &index) 
    {
        std::cout << "Ending process" << std::endl;
    }
    
    void handleProcess(size_t& index) {
        if(index+2<tokens.size()) { std::cout<<"Processing: "<<tokens[index+1]<<" "<<tokens[index+2]<<std::endl; index+=2; }
    }
};

int main(int argc, char *argv[]) {
    std::string testInput = "START : process data1 , data2 ; END";
    Parser* parser = new Parser(testInput);
    
    if(parser->parse()) {
        std::cout << "Parse successful!" << std::endl;
        parser->processCommands();
    } else { std::cerr<<"Parse failed"<<std::endl; }
    
    delete parser;
    return 0;
}