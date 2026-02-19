import Foundation
import Alamofire
import SwiftyJSON

class NetworkManager {
    static let shared = NetworkManager()
    
    private init() {}
    
    // Fetch weather data from API using Alamofire
    func fetchWeatherData(for city: String, completion: @escaping (Result<JSON, Error>) -> Void) {
        let urlString = "https://api.weather.com/data/\(city)"
        
        // TODO: Implement Alamofire request
        // AF.request(urlString).responseJSON { response in
        //     Parse response with SwiftyJSON
        // }
    }
    
    // Fetch forecast data using Alamofire
    func fetchForecast(latitude: Double, longitude: Double, completion: @escaping (Result<JSON, Error>) -> Void) {
        let parameters: [String: Any] = ["lat": latitude, "lon": longitude]
        
        // TODO: Implement Alamofire request with parameters
        // TODO: Convert response to SwiftyJSON object for easy parsing
    }
    
    // Generic API request helper using Alamofire
    func performRequest(endpoint: String, method: HTTPMethod = .get, completion: @escaping (Result<JSON, Error>) -> Void) {
        // TODO: Use Alamofire to perform HTTP request
        // TODO: Parse JSON response with SwiftyJSON
        // TODO: Handle errors appropriately
    }
}