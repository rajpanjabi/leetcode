### Structural design patterns

Structural design patterns are concerned with the composition of classes and objects. They focus on how to assemble classes and objects into larger structures while keeping these structures flexible and efficient. Adapter Pattern is one of the most important structural design patterns. Let's understand in depth.




Adaptor design pattern is used when you have a class that has similar functionality but cannot be used because of inconsistent structure. For instance, consider the scenario of PaymentGateway, we have an payment interface that is implemented by other payment providers, and an order class that needs a paymentgateway. If we had another class that is a payment provider but doesn't implement the interface and has inconsistent structure in adherance to the payment interface, we can use Adaptor pattern.

We create a new adaptee class that implements the gateway interface, we also add the concrete class as a dependency in this class and initialise it whenever this adaptee class's instance is created. Now in the method, we call the method of the class from the method of adaptee class.

Adaptor pattern is mostly used for Payment Gateways, Logging frameworks, Cloud Providers and SDKs

