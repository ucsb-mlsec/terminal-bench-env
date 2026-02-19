import UIKit
import Alamofire
import SwiftyJSON
import Kingfisher
import SDWebImage
import SnapKit

class ViewController: UIViewController {
    
    private let weatherImageView = UIImageView()
    private let temperatureLabel = UILabel()
    private let cityLabel = UILabel()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        fetchWeatherData()
    }
    
    func fetchWeatherData() {
        // TODO: Use Alamofire for networking
        // Example: AF.request("https://api.weather.com/data").response { response in }
        
        // TODO: Parse response with SwiftyJSON
        // Example: let json = JSON(response.data)
    }
    
    func setupUI() {
        view.backgroundColor = .white
        
        // TODO: Use SnapKit for constraints
        // Example: weatherImageView.snp.makeConstraints { make in }
        
        view.addSubview(weatherImageView)
        view.addSubview(temperatureLabel)
        view.addSubview(cityLabel)
    }
    
    func loadWeatherImage(from url: String) {
        // TODO: Use Kingfisher or SDWebImage to load images
        // Example: weatherImageView.kf.setImage(with: URL(string: url))
        // Or: weatherImageView.sd_setImage(with: URL(string: url))
    }
}