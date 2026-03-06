import Alamofire
import SwiftyJSON
import Vapor

class NetworkManager {
    static let shared = NetworkManager()
    
    func fetchData(from url: String) {
        AF.request(url).responseJSON { response in
            switch response.result {
            case .success(let value):
                let json = JSON(value)
                self.parseResponse(json)
            case .failure(let error):
                print("Error: \(error)")
            }
        }
    }
    
    func parseResponse(_ json: JSON) {
        let items = json["items"].arrayValue
        print("Found \(items.count) items")
    }
}

let manager = NetworkManager.shared
manager.fetchData(from: "https://api.example.com/data")