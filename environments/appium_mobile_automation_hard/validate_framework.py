#!/usr/bin/env python3

import json
import yaml
import os
import sys
import re

def validate_config_files():
    """Validate that config files exist and contain required keys."""
    try:
        # Check appium_config.json
        config_path = 'config/appium_config.json'
        if not os.path.exists(config_path):
            print(f"Error: {config_path} does not exist")
            return False
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        if 'server' not in config_data or 'capabilities' not in config_data:
            print("Error: appium_config.json missing required keys 'server' or 'capabilities'")
            return False
        
        # Check test_suite.yaml
        suite_path = 'config/test_suite.yaml'
        if not os.path.exists(suite_path):
            print(f"Error: {suite_path} does not exist")
            return False
        
        with open(suite_path, 'r') as f:
            suite_data = yaml.safe_load(f)
        
        if 'suite_name' not in suite_data or 'test_scenarios' not in suite_data:
            print("Error: test_suite.yaml missing required keys 'suite_name' or 'test_scenarios'")
            return False
        
        print("✓ Config files validation passed")
        return True
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in appium_config.json - {e}")
        return False
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in test_suite.yaml - {e}")
        return False
    except Exception as e:
        print(f"Error validating config files: {e}")
        return False

def validate_capabilities():
    """Validate Appium capabilities for Android."""
    try:
        config_path = 'config/appium_config.json'
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        capabilities = config_data.get('capabilities', {})
        
        # Check platformName
        if capabilities.get('platformName') != 'Android':
            print("Error: platformName must be 'Android' (case-sensitive)")
            return False
        
        # Check platformVersion
        if 'platformVersion' not in capabilities or not capabilities['platformVersion']:
            print("Error: platformVersion must be present and non-empty")
            return False
        
        # Check deviceName
        if 'deviceName' not in capabilities:
            print("Error: deviceName must be present")
            return False
        
        # Check automationName
        if capabilities.get('automationName') != 'UiAutomator2':
            print("Error: automationName must be 'UiAutomator2'")
            return False
        
        print("✓ Capabilities validation passed")
        return True
        
    except Exception as e:
        print(f"Error validating capabilities: {e}")
        return False

def validate_dependencies():
    """Validate dependencies.json for correct versions."""
    try:
        deps_path = 'dependencies.json'
        if not os.path.exists(deps_path):
            print(f"Error: {deps_path} does not exist")
            return False
        
        with open(deps_path, 'r') as f:
            deps_data = json.load(f)
        
        # Check Appium-Python-Client version (must be 2.x.x)
        appium_version = deps_data.get('Appium-Python-Client', '')
        if not re.match(r'^2\.\d+\.\d+$', appium_version):
            print(f"Error: Appium-Python-Client version must be '2.x.x' format, got '{appium_version}'")
            return False
        
        # Check selenium version (should be 4.x.x)
        selenium_version = deps_data.get('selenium', '')
        if not re.match(r'^4\.\d+\.\d+$', selenium_version):
            print(f"Error: selenium version should be '4.x.x' format, got '{selenium_version}'")
            return False
        
        # Check python_version (>= 3.8)
        python_version = deps_data.get('python_version', '')
        if python_version:
            version_match = re.match(r'^(\d+)\.(\d+)', python_version)
            if version_match:
                major, minor = int(version_match.group(1)), int(version_match.group(2))
                if major < 3 or (major == 3 and minor < 8):
                    print(f"Error: python_version should be >= '3.8', got '{python_version}'")
                    return False
            else:
                print(f"Error: Invalid python_version format: '{python_version}'")
                return False
        
        print("✓ Dependencies validation passed")
        return True
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in dependencies.json - {e}")
        return False
    except Exception as e:
        print(f"Error validating dependencies: {e}")
        return False

def validate_tests():
    """Validate test_suite.yaml structure."""
    try:
        suite_path = 'config/test_suite.yaml'
        with open(suite_path, 'r') as f:
            suite_data = yaml.safe_load(f)
        
        test_scenarios = suite_data.get('test_scenarios', [])
        
        if not isinstance(test_scenarios, list):
            print("Error: test_scenarios must be a list")
            return False
        
        for idx, scenario in enumerate(test_scenarios):
            # Check 'name' field
            if 'name' not in scenario:
                print(f"Error: test_scenario {idx} missing 'name' field")
                return False
            
            # Check 'file' field
            if 'file' not in scenario:
                print(f"Error: test_scenario {idx} missing 'file' field")
                return False
            
            # Check 'enabled' is boolean
            if 'enabled' in scenario and not isinstance(scenario['enabled'], bool):
                print(f"Error: test_scenario {idx} 'enabled' must be boolean")
                return False
            
            # Check 'priority' is valid
            if 'priority' in scenario:
                if scenario['priority'] not in ['high', 'medium', 'low']:
                    print(f"Error: test_scenario {idx} 'priority' must be one of ['high', 'medium', 'low']")
                    return False
        
        print("✓ Tests validation passed")
        return True
        
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in test_suite.yaml - {e}")
        return False
    except Exception as e:
        print(f"Error validating tests: {e}")
        return False

if __name__ == '__main__':
    print("Starting Appium framework validation...\n")
    
    # Run all validations
    config_valid = validate_config_files()
    capabilities_valid = validate_capabilities()
    dependencies_valid = validate_dependencies()
    tests_valid = validate_tests()
    
    # Print summary
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    print(f"Config Files: {'PASS' if config_valid else 'FAIL'}")
    print(f"Capabilities: {'PASS' if capabilities_valid else 'FAIL'}")
    print(f"Dependencies: {'PASS' if dependencies_valid else 'FAIL'}")
    print(f"Tests: {'PASS' if tests_valid else 'FAIL'}")
    print("="*50)
    
    # Write results to file
    result_path = '/workspace/validation_result.txt'
    try:
        with open(result_path, 'w') as f:
            f.write(f"config_valid={'true' if config_valid else 'false'}\n")
            f.write(f"capabilities_valid={'true' if capabilities_valid else 'false'}\n")
            f.write(f"dependencies_valid={'true' if dependencies_valid else 'false'}\n")
            f.write(f"tests_valid={'true' if tests_valid else 'false'}\n")
        print(f"\nValidation results written to {result_path}")
    except Exception as e:
        print(f"Error writing validation results: {e}")
        sys.exit(1)
    
    # Exit with appropriate code
    if all([config_valid, capabilities_valid, dependencies_valid, tests_valid]):
        print("\n✓ All validations passed!")
        sys.exit(0)
    else:
        print("\n✗ Some validations failed. Please fix the issues above.")
        sys.exit(1)