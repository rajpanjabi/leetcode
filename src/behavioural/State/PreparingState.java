package src.behavioural.State;

public class PreparingState implements OrderState {
    @Override
    public void next(OrderContext order){
        order.setState(new OutForDeliveryState());
        System.out.println("Order is Out for Delivery");
    }
    @Override
    public void cancel(OrderContext order){
        order.setState(new CancelledState());
        System.out.println("Cancelling Order");
    }
    @Override
    public String getStateName(){
        return "PREPARING_STATE";
    }
}