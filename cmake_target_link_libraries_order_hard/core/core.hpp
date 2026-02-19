#pragma once

#include <string>

namespace Core {

class CoreBase {
public:
    virtual ~CoreBase() = default;
    virtual void execute() = 0;
    virtual std::string getName() const = 0;
};

std::string getCoreVersion();

void initializeCore();

} // namespace Core