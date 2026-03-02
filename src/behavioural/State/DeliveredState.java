package src.behavioural.State;

public class DeliveredState implements OrderState{
    @Override
    public void next(OrderContext order){
        System.out.println("Order is already delivered.");
    }
    @Override
    public void cancel(OrderContext order){
        System.out.println("Cannot be cancelled, already delivered");
    }
    @Override
    public String getStateName(){
        return "DELIVERED";
    }
}