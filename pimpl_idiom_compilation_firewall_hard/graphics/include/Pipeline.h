#ifndef PIPELINE_H
#define PIPELINE_H

#include <vector>
#include <string>
#include <memory>

class Pipeline {
public:
    Pipeline();
    ~Pipeline();
    void addStage(const std::string& name);
    void execute();
    int getStageCount() const;

private:
    std::vector<std::string> m_stages;
    std::string m_name;
    std::unique_ptr<int> m_config;
};

#endif