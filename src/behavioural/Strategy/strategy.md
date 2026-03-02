Strategy pattern helps us when we want to change how a class behaves at runtime. 

More formally, Strategy Pattern is a behavioral design pattern that defines a family of algorithms, encapsulates each one into a separate class, and makes them interchangeable at runtime depending on the context.

Let's consider example of Uber's ride matching service. This service can use different types of matching strategies like nearestDriver, surgePriority, airportFIFO, and so on..

If we had simple logic of if and else inside of the ride matching service class, it would violate the SRP because it handles logic of ride match and also of creatign new strategy. It would also vioalte OCP, if in future we wanted to add a new strategy.

So, we use strategy pattern in this case, we have an interface MatchingStrategy with one method match. We have concrete classes of matching strategies like nearestDriver, surgePriority, airportFIFO that implement this interface. We have the ride matching service class that is initialised with the type of ride matching strategy, so it follows srp. Now for the matchRider method it uses the match method of the matching strategy used to initialise the service class.

