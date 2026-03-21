package structural.Adaptor;

public class Stripe implements PaymentGateway {
    @Override
    public void pay(String orderId, double amount){
            System.out.println("Paid Rs. " + amount + " using Stripe for order: " + orderId);
    }   
}
