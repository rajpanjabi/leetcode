package src.behavioural.State;

public class OrderContext {
    private OrderState currentState;

    public OrderContext(){
        this.currentState= new OrderPlacedState();
    }
    // Order should have cancel, go to next and methods which are 
    // implemented by the respective methods of that state

    public void setState(OrderState state){
        this.currentState=state;
    }
    public void next(){
        currentState.next(this);
    }
    public void cancel(){
        currentState.cancel(this);
    }
    public String getOrderState(){
        return currentState.getStateName();
    }


}
