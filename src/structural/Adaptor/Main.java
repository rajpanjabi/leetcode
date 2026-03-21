package structural.Adaptor;

public class Main {
    
    public static void main(String[] args) {
        // Here we create two different OrderService service and check if they work as expected
        OrderService service1 =new OrderService(new Stripe());
        service1.processOrder("123", 250);
        // second service using the adaptor
        OrderService service2 =new OrderService(new PayPalAdaptor());
        service2.processOrder("456", 520);

    }
}
