#include <rclcpp/rclcpp.hpp>
#include "higgs_rc/nodes/higgsr_server_node.hpp"

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    
    rclcpp::NodeOptions options;
    options.allow_undeclared_parameters(true);
    options.automatically_declare_parameters_from_overrides(true);
    
    auto node = std::make_shared<higgs_rc::nodes::HiggsrServerNode>(options);
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    
    return 0;
}