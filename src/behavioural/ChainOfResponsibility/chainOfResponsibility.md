## Chain of Responsibility Pattern

Chain of Responsibility Pattern, allows a request to be passed along a chain of potential handlers until one of them processes it. This pattern is particularly useful for scenarios where multiple handlers might be responsible for processing a request, and we want to avoid tightly coupling the sender of the request to the receivers.

Imagine you're building a customer support system with different levels of support, such as basic inquiries, technical issues, and advanced troubleshooting. A customer can send in a request, and depending on its complexity, the request should be forwarded to the appropriate team. Instead of each team individually checking if they can handle every possible request, the Chain of Responsibility Pattern sets up a chain where each team can either process the request or pass it to the next team in the chain. This enables a flexible and extensible system where adding new handlers (teams) is easy and doesn't require changes to the existing code.


## Key Components
This pattern consists of the following components:
- Handler: An abstract class or interface that defines the method for handling requests and a reference to the next handler in the chain.
- Concrete Handler: A class that implements the handler and processes the request if it can. Otherwise, it forwards the request to the next handler.
- Client: The object that sends the request, typically unaware of the specific handler that will process it.

## Solution without Chain of Responsibility

Problems:
- Monolithic code
- Violates OCP when new handler needs to be introduced
-

```java
import java.util.*;

// SupportService class: Handles different types of support requests
class SupportService {

    // Method to handle the support request based on the type of issue
    public void handleRequest(String type) {
        if (type.equals("general")) {
            System.out.println("Handled by General Support");
        } else if (type.equals("refund")) {
            System.out.println("Handled by Billing Team");
        } else if (type.equals("technical")) {
            System.out.println("Handled by Technical Support");
        } else if (type.equals("delivery")) {
            System.out.println("Handled by Delivery Team");
        } else {
            System.out.println("No handler available");
        }
    }
}

// Main class: Entry point to test the chain of responsibility pattern
public class Main {

    public static void main(String[] args) {
        // Create an instance of SupportService
        SupportService supportService = new SupportService();
        
        // Test with different types of requests
        supportService.handleRequest("general");
        supportService.handleRequest("refund");
        supportService.handleRequest("technical");
        supportService.handleRequest("delivery");
        supportService.handleRequest("unknown");
    }
}
```