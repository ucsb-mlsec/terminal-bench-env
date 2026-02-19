#!/usr/bin/env python3

import unittest
import time
import sys
import os

# Add the plugin_system directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plugin_system


class TestLazyLoading(unittest.TestCase):
    
    def test_plugins_not_initialized_at_startup(self):
        """Verify that plugins have not been initialized after importing"""
        # Check that plugin instances don't have initialized attribute set to True yet
        plugin_names = ['database', 'api', 'cache', 'logging', 'analytics', 'search', 'notification', 'auth']
        
        for plugin_name in plugin_names:
            # The plugin should exist as an attribute but not be initialized
            self.assertTrue(hasattr(plugin_system, plugin_name), 
                          f"Plugin {plugin_name} should exist")
            
            plugin = getattr(plugin_system, plugin_name)
            
            # Check if it's a lazy-loaded object (not yet initialized)
            # It should either not have an 'initialized' attribute or it should be False
            if hasattr(plugin, 'initialized'):
                self.assertFalse(plugin.initialized, 
                               f"Plugin {plugin_name} should not be initialized at startup")
    
    def test_plugin_initialized_on_first_access(self):
        """Access a plugin and verify it gets initialized"""
        # Access the database plugin
        db_plugin = plugin_system.database
        
        # Try to access a method or attribute that would trigger initialization
        # This should initialize the plugin
        result = db_plugin.query("SELECT * FROM test")
        
        # Now check that it's initialized
        self.assertTrue(hasattr(db_plugin, 'initialized'), 
                       "Plugin should have initialized attribute after access")
        self.assertTrue(db_plugin.initialized, 
                       "Plugin should be initialized after first access")
        
        # Verify the functionality works
        self.assertIsNotNone(result, "Plugin method should return a result")
    
    def test_plugin_reused_on_subsequent_access(self):
        """Access same plugin twice and verify it's the same instance"""
        # Access the api plugin for the first time
        api_plugin_first = plugin_system.api
        first_result = api_plugin_first.call_api("/test")
        
        # Get the object id of the first access
        first_id = id(api_plugin_first)
        
        # Access the same plugin again
        api_plugin_second = plugin_system.api
        second_result = api_plugin_second.call_api("/test")
        
        # Get the object id of the second access
        second_id = id(api_plugin_second)
        
        # They should be the same instance
        self.assertEqual(first_id, second_id, 
                        "Plugin should be reused on subsequent access")
        
        # Both should be initialized
        self.assertTrue(api_plugin_first.initialized)
        self.assertTrue(api_plugin_second.initialized)
    
    def test_all_plugins_work(self):
        """Test that all 8 plugins work correctly by calling their methods"""
        plugin_tests = [
            ('database', 'query', ["SELECT * FROM test"]),
            ('api', 'call_api', ["/endpoint"]),
            ('cache', 'get', ["key"]),
            ('logging', 'log', ["message"]),
            ('analytics', 'track', ["event"]),
            ('search', 'search', ["query"]),
            ('notification', 'send', ["message"]),
            ('auth', 'authenticate', ["user", "pass"])
        ]
        
        for plugin_name, method_name, args in plugin_tests:
            with self.subTest(plugin=plugin_name):
                plugin = getattr(plugin_system, plugin_name)
                self.assertTrue(hasattr(plugin, method_name), 
                              f"Plugin {plugin_name} should have method {method_name}")
                
                method = getattr(plugin, method_name)
                result = method(*args)
                
                # Verify the plugin is now initialized
                self.assertTrue(plugin.initialized, 
                              f"Plugin {plugin_name} should be initialized after use")
                
                # Verify the method returns something
                self.assertIsNotNone(result, 
                                   f"Plugin {plugin_name}.{method_name} should return a result")
    
    def test_startup_time_improvement(self):
        """Measure startup time and verify >= 90% improvement"""
        baseline_file = '/workspace/plugin_system/baseline_time.txt'
        
        # Read baseline time
        self.assertTrue(os.path.exists(baseline_file), 
                       "Baseline time file should exist")
        
        with open(baseline_file, 'r') as f:
            baseline_time = float(f.read().strip())
        
        # Measure current startup time by reimporting the module
        if 'plugin_system' in sys.modules:
            del sys.modules['plugin_system']
        
        start_time = time.time()
        import plugin_system as ps
        end_time = time.time()
        
        startup_time_ms = (end_time - start_time) * 1000
        
        # Calculate improvement percentage
        improvement_percent = ((baseline_time - startup_time_ms) / baseline_time) * 100
        
        # Store for later use
        self.startup_time_ms = startup_time_ms
        self.improvement_percent = improvement_percent
        
        # Assert >= 90% improvement
        self.assertGreaterEqual(improvement_percent, 90, 
                               f"Startup time improvement should be >= 90%, got {improvement_percent:.2f}%")
    
    def test_functionality_preserved(self):
        """Test specific plugin methods work as expected"""
        # Test DatabasePlugin.query
        db_result = plugin_system.database.query("SELECT * FROM users")
        self.assertIn("result", str(db_result).lower())
        
        # Test APIPlugin.call_api
        api_result = plugin_system.api.call_api("/users/123")
        self.assertIsNotNone(api_result)
        
        # Test CachePlugin.get and set
        plugin_system.cache.set("test_key", "test_value")
        cache_result = plugin_system.cache.get("test_key")
        self.assertEqual(cache_result, "test_value")
        
        # Test LoggingPlugin.log
        log_result = plugin_system.logging.log("Test log message")
        self.assertIsNotNone(log_result)
        
        # Test AnalyticsPlugin.track
        analytics_result = plugin_system.analytics.track("page_view")
        self.assertIsNotNone(analytics_result)
        
        # Test SearchPlugin.search
        search_result = plugin_system.search.search("test query")
        self.assertIsNotNone(search_result)
        
        # Test NotificationPlugin.send
        notif_result = plugin_system.notification.send("Test notification")
        self.assertIsNotNone(notif_result)
        
        # Test AuthPlugin.authenticate
        auth_result = plugin_system.auth.authenticate("admin", "password123")
        self.assertIsNotNone(auth_result)


if __name__ == '__main__':
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestLazyLoading)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Write solution.txt if test_startup_time_improvement ran
    test_instance = TestLazyLoading()
    
    # Run startup time test separately to capture metrics
    startup_suite = unittest.TestSuite()
    startup_suite.addTest(TestLazyLoading('test_startup_time_improvement'))
    startup_runner = unittest.TextTestRunner(verbosity=0)
    startup_result = startup_runner.run(startup_suite)
    
    # Get the test instance that ran
    for test in startup_suite:
        if hasattr(test, 'startup_time_ms'):
            startup_time_ms = int(test.startup_time_ms)
            improvement_percent = int(test.improvement_percent)
            break
    else:
        # Fallback: measure it ourselves
        baseline_file = '/workspace/plugin_system/baseline_time.txt'
        if os.path.exists(baseline_file):
            with open(baseline_file, 'r') as f:
                baseline_time = float(f.read().strip())
            
            if 'plugin_system' in sys.modules:
                del sys.modules['plugin_system']
            
            start_time = time.time()
            import plugin_system as ps
            end_time = time.time()
            
            startup_time_ms = int((end_time - start_time) * 1000)
            improvement_percent = int(((baseline_time - startup_time_ms) / baseline_time) * 100)
        else:
            startup_time_ms = 0
            improvement_percent = 0
    
    all_tests_passed = result.wasSuccessful()
    
    # Write solution file
    solution_path = '/workspace/solution.txt'
    with open(solution_path, 'w') as f:
        f.write(f"startup_time_ms={startup_time_ms}\n")
        f.write(f"improvement_percent={improvement_percent}\n")
        f.write(f"all_tests_passed={str(all_tests_passed).lower()}\n")
    
    print(f"\nSolution written to {solution_path}")
    print(f"Startup time: {startup_time_ms}ms")
    print(f"Improvement: {improvement_percent}%")
    print(f"All tests passed: {all_tests_passed}")