package structural.Adaptor;

interface PaymentGateway {
    void pay(String orderId, double amount);
}
