#include "Pipeline.h"
#include <vector>
#include <string>
#include <memory>

struct Pipeline::Impl {
    std::vector<std::string> m_stages;
    std::string m_name;
    std::unique_ptr<int> m_config;
    
    Impl() : m_stages(), m_name("pipeline"), m_config(new int(0)) {}
};

Pipeline::Pipeline() : pImpl(new Impl()) {
}

Pipeline::~Pipeline() {
    delete pImpl;
}

void Pipeline::addStage(const std::string& name) {
    pImpl->m_stages.push_back(name);
}

void Pipeline::execute() {
    // Execution logic would go here
}

int Pipeline::getStageCount() const {
    return pImpl->m_stages.size();
}