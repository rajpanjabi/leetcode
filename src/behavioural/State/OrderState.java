package src.behavioural.State;

public interface OrderState {
    void next(OrderContext order);
    void cancel(OrderContext order);
    String getStateName();
   
}
