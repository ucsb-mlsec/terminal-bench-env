import Foundation
import SwiftyJSON

struct DataParser {
    
    private let decoder = JSONDecoder()
    
    func parseJSON(data: Data) -> JSON? {
        return try? JSON(data: data)
    }
    
    func decodeData<T: Decodable>(_ type: T.Type, from data: Data) -> T? {
        return try? decoder.decode(type, from: data)
    }
    
    func extractValue(from json: JSON, key: String) -> String? {
        return json[key].string
    }
    
    func parseArray(data: Data) -> [JSON]? {
        guard let json = parseJSON(data: data) else { return nil }
        return json.array
    }
    
    func isValidJSON(data: Data) -> Bool {
        return parseJSON(data: data) != nil
    }
}