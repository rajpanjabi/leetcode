package src.creational.AbstractFactory;

public class RazorPayGateway implements PaymentGateway {
    @Override
    public void processPayment(double amount){
        System.out.println("RazorPay processing payment "+ amount);
    }
}
