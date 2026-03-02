package src.behavioural.State;

public class OrderPlacedState implements OrderState{
    
    @Override
    public void next(OrderContext order){
        order.setState(new PreparingState());
        System.out.println("Order is now being prepared");
    }

    @Override
    public void cancel(OrderContext order){
        System.out.println("Cancelling order");
        order.setState(new CancelledState());
    }

    @Override
    public String getStateName(){
        return "ORDER_PLACED";
    }
}
