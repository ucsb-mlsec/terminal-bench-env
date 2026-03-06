import Foundation
import Alamofire
import SwiftyJSON

class NetworkManager {
    var baseURL: String
    var timeout: TimeInterval
    
    init(baseURL: String, timeout: TimeInterval = 30.0) {
        self.baseURL = baseURL
        self.timeout = timeout
    }
    
    func fetchData(endpoint: String, completion: @escaping (Result<Data, Error>) -> Void) {
        let url = baseURL + endpoint
        AF.request(url, method: .get).response { response in
            if let data = response.data {
                completion(.success(data))
            } else if let error = response.error {
                completion(.failure(error))
            }
        }
    }
    
    func postData(endpoint: String, parameters: [String: Any], completion: @escaping (Bool) -> Void) {
        let url = baseURL + endpoint
        AF.request(url, method: .post, parameters: parameters).response { response in
            completion(response.error == nil)
        }
    }
    
    func parseJSON(data: Data) -> JSON? {
        return try? JSON(data: data)
    }
}