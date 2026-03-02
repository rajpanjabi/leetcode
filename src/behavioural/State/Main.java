package src.behavioural.State;

public class Main {
    
    public static void main(String[] args) {
        // Creating an order using OrderContext class
        OrderContext order = new OrderContext();
        System.out.println(order.getOrderState());
        order.next();
        System.out.println(order.getOrderState());
        order.next();
        System.out.println(order.getOrderState());
       
        order.cancel();
    }
}
