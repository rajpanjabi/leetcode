package src.creational.AbstractFactory;

public class PayUGateway implements PaymentGateway{
    @Override
    public void processPayment(double amount){
        System.out.println("PayU processing payment "+ amount);
    }
}

