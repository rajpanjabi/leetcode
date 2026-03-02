package src.behavioural.State;

public class OutForDeliveryState implements OrderState{
    @Override
    public void next(OrderContext order){
        order.setState(new DeliveredState());
        System.out.println("Order has been delivered");
    }
    @Override
    public void cancel(OrderContext order){
        System.out.println("Cannot be cancelled, out for delivery");
    }
    @Override
    public String getStateName(){
        return "Out_For_Delivery";
    }
   
}
