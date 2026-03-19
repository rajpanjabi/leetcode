
The Template Pattern is a behavioral design pattern that provides a blueprint for executing an algorithm. It allows subclasses to override specific steps of the algorithm, but the overall structure remains the same. This ensures that the invariant parts of the algorithm are not changed, while enabling customization in the variable parts.

Imagine you are following a recipe to bake a cake. The overall process of baking a cake (preheat oven, mix ingredients, bake, and cool) is fixed, but the specific ingredients or flavors may vary (chocolate, vanilla, etc.).


Key steps in Template Pattern:

1) A final template method: this is a final method that follows a fixed sequence of methods/steps. It calls the abstract (varying logic implemented by specific subclass), non-abstract(common behaviour methods implemented in the base abstract class) and optionally default methods. This is final method and hence cannot be overriden by subclass, and hence no change can be made to the seq of methods.
2) Abstract methods: These are the methods that needs to be implemented by subclasses with the specific logic (different for red-velvet, blackforest and so on). 
3) Concrete (final or concrete methods): This is the common repeating process/behaviour to all subclasses. They are defined in the base class and shared by all subclasses.
4) Hooks (Optional methods with Defualt behaviour): Hooks are optional methods in the base class that provide default behaviour. Subclasses can override these methods to modify the behaviour when needed, but they are not mandatory for all subclasses to implement.

By using the Template Pattern, one can ensure that the common steps of an algorithm remain unchanged while allowing subclasses to modify the specific details of the algorithm.


Example of when we should use template method.

Let's say we have a Notification system that uses SMS and Email to send notification, both have send method which has most of the steps common/same and minor changes to the core logic.

```java
class EmailNotification{
    public void send(String message){
        System.out.println("Checking rate limits for: " + to);
        System.out.println("Validating email recipient: " + to);
        String formatted = message.trim();
        System.out.println("Logging before send: " + formatted + " to " + to);

        // Compose Email
        String composedMessage = "<html><body><p>" + formatted + "</p></body></html>";

        // Send Email
        System.out.println("Sending EMAIL to " + to + " with content:\n" + composedMessage);

        // Analytics
        System.out.println("Analytics updated for: " + to);
    
    }

}


class SMSNotification{
    public void send(String message){
        System.out.println("Checking rate limits for: " + to);
        System.out.println("Validating phone number: " + to);
        String formatted = message.trim();
        System.out.println("Logging before send: " + formatted + " to " + to);

        // Compose SMS
        String composedMessage = "[SMS]" + formatted ;

        // Send SMS
        System.out.println("Sending EMAIL SMS " + to + " with content:\n" + composedMessage);

        // Analytics
        System.out.println("Analytics updated for: " + to);
    
    }

}

class Main {
    public static void main(String[] args) {
        // Create objects for both notification services
        EmailNotification emailNotification = new EmailNotification();
        SMSNotification smsNotification = new SMSNotification();

        // Sending email notification
        emailNotification.send("example@example.com", "Your order has been placed!");
        
        System.out.println(" ");
        
        // Sending SMS notification
        smsNotification.send("1234567890", "Your OTP is 1234.");
    }
}
```
