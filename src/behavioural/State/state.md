

The State Pattern is a behavioral design pattern that encapsulates state-specific behavior into separate classes and delegates the behavior to the appropriate state object. This allows the object to change its behavior without altering the underlying code.


This pattern makes it easy to manage state transitions by isolating state-specific behavior into distinct classes. It helps achieve loose coupling by ensuring that each state class is independent and can evolve without affecting others.


Let's take an example of food delivery app. As an order progresses
, its state changes through multiple stages:
Order Placed -> Preparing -> Delivery partner assigned -> Order picked up -> out for delivery -> order delivered

So, we isolate each of the state in a different class. Here states depend on each other, when one finishes then only we move to the next one. Each state stores logic for the respective state and also what the next state would be, it also has logic of handling cancellation of order.

At each stage, the app behaves differently:
- In the "Order Placed" state, you can cancel the order.
- In the "Order Preparing" state, you can track the preparation status.
- In the "Delivery Partner Assigned" state, you can see the details of the assigned driver.
- And so on until the order is delivered.

Naive implemenatation of problem at hand without State Pattern

```java

class Order{
    public String state;
    public Order(){
        this.state="ORDER_PLACED";
    }

    // Method to cancel the order 
    // only allows cancellation if in ORDER_PLACED or PREPARING states
    public void cancelOrder() {
        if (state.equals("ORDER_PLACED") || state.equals("PREPARING")) {
            state = "CANCELLED";
            System.out.println("Order has been cancelled.");
        } else {
            System.out.println("Cannot cancel the order now.");
        }
    }



    public void next_state(){
        switch(state){
            case "ORDER_PLACED":
                state="PREPARING";
                break;
            case "PREPARING":
                state="OUT_FOR_DELIVERY";
                break;
            case "OUT_FOR_DELIVERY":
                state="DELIVERED";
                break;
            default:
                System.out.println("No next state from: " + state);
                return;
        } 
        System.out.println("Order moved to: " + state);
    }
    public String getState(){
        return state
    }
}
class Main{
    public static void main(String[] args){
        // Create an order
        Order order =new Order();
        // init state
        System.out.println("Initial State: " + order.getState());
        order.next_state(); // ORDER_PLACED -> PREPARING
        order.next_state(); // PREPARING -> OUT_FOR_DELIVERY
        order.next_state(); // OUT_FOR_DELIVERY -> DELIVERED

        // Attempting to cancel an order after it is out for delivery
        order.cancelOrder(); // Should not allow cancellation

        // Display final state
        System.out.println("Final State: " + order.getState());

    }    

}
```


Here there are lot of things happening in one class, also the code doesnt follow SRP, OCP i.e it is not doing just one task and neither it is open for extension.

So, we use State pattern to fix these issues and make code modular, scalable.

We create an OrderState interface which is implemented by concrete classes of all of the states. This has the logic for each of the respective state and we have an OrderContext class which holds the main order object and the current state.