package src.behavioural.State;

public class CancelledState implements OrderState {
    @Override
    public void next(OrderContext context) {
        System.out.println("Cancelled order cannot move to next state.");
    }
    @Override
    public void cancel(OrderContext context) {
        System.out.println("Order is already cancelled.");
    }
    @Override
    public String getStateName() {
        return "CANCELLED";
    }
}