package simulations

import io.gatling.core.Predef._
import io.gatling.http.Predef._
import scala.concurrent.duration._

class CheckoutSimulation extends Simulation {

  // HTTP Protocol Configuration
  val httpProtocol = http
    .baseUrl("http://example-shop.com")
    .acceptHeader("application/json")
    .acceptEncodingHeader("gzip, deflate")
    .acceptLanguageHeader("en-US,en;q=0.9")
    .userAgentHeader("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

  // Scenario Definition
  val checkoutScenario = scenario("E-Commerce Checkout Flow")
    .exec(http("Browse Products")
      .get("/api/products")
      .check(status.is(200)))
    .pause(2.seconds)
    
    .exec(http("View Product Details")
      .get("/api/products/42")
      .check(status.is(200))
      .check(jsonPath("$.id").exists))
    .pause(3.seconds)
    
    .exec(http("Add to Cart")
      .post("/api/cart")
      .header("Content-Type", "application/json")
      .body(StringBody("""{"productId": 42, "quantity": 1}"""))
      .check(status.is(201)))
    .pause(1.second)
    
    .exec(http("View Cart")
      .get("/api/cart")
      .check(status.is(200))
      .check(jsonPath("$.items[0].productId").is("42")))
    .pause(2.seconds)
    
    .exec(http("Proceed to Checkout")
      .post("/api/checkout")
      .header("Content-Type", "application/json")
      .body(StringBody("""{"paymentMethod": "credit_card", "shippingAddress": "123 Main St"}"""))
      .check(status.is(200))
      .check(jsonPath("$.orderId").exists))

  // Load Injection Profile
  setUp(
    checkoutScenario.inject(
      atOnceUsers(10),
      rampUsers(50).during(30.seconds)
    )
  ).protocols(httpProtocol)
    .assertions(
      global.responseTime.max.lt(5000),
      global.successfulRequests.percent.gt(95)
    )
}