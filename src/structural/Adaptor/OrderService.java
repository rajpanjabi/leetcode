package structural.Adaptor;

public class OrderService {
    // the order service class uses the Payment method from a payment gateway class, since we want the code 
    // to be extensible, we use an interface here and make the selection of the gateway dynamic.
    PaymentGateway gateway;

    public OrderService(PaymentGateway gateway){
        this.gateway=gateway;
    }

    public void processOrder(String orderId, double amount){
        System.out.println("Processing order"+ orderId);
        gateway.pay(orderId, amount);
    }
}
