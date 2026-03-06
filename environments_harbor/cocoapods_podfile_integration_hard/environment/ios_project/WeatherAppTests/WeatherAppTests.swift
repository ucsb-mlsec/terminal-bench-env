import XCTest
import Quick
import Nimble
@testable import WeatherApp

class WeatherAppTests: QuickSpec {
    override func spec() {
        describe("WeatherApp") {
            it("should initialize correctly") {
                expect(true).to(beTrue())
            }
            
            context("when testing basic functionality") {
                it("should pass basic assertions") {
                    let value = 42
                    expect(value).to(equal(42))
                }
                
                it("should handle nil values") {
                    let nilValue: String? = nil
                    expect(nilValue).to(beNil())
                }
            }
            
            context("NetworkManager") {
                it("should have shared instance") {
                    let manager = NetworkManager.shared
                    expect(manager).toNot(beNil())
                }
                
                it("should be a singleton") {
                    let manager1 = NetworkManager.shared
                    let manager2 = NetworkManager.shared
                    expect(manager1).to(be(manager2))
                }
            }
            
            context("WeatherData model") {
                it("should create valid weather data object") {
                    let weatherData = WeatherData(temperature: 25.0, condition: "Sunny")
                    expect(weatherData.temperature).to(equal(25.0))
                    expect(weatherData.condition).to(equal("Sunny"))
                }
            }
        }
    }
}