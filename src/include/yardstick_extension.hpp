#pragma once

#include "duckdb.hpp"

namespace duckdb {

// Main extension class
class YardstickExtension : public Extension {
public:
    void Load(ExtensionLoader &loader) override;
    std::string Name() override { return "yardstick"; }
    std::string Version() const override;
};

} // namespace duckdb
