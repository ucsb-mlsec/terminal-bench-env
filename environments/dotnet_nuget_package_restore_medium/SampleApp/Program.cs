using System;
using System.Collections.Generic;
using Newtonsoft.Json;

namespace SampleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Application started successfully");
            
            // Simple usage of Newtonsoft.Json to demonstrate package dependency
            string jsonString = "{\"message\":\"Hello World\",\"count\":42}";
            var data = JsonConvert.DeserializeObject<Dictionary<string, object>>(jsonString);
            
            Console.WriteLine($"Deserialized message: {data["message"]}");
            Console.WriteLine("Build verification complete");
        }
    }
}