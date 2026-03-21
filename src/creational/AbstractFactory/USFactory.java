package src.creational.AbstractFactory;

public class USFactory implements RegionFactory{
    @Override
    public PaymentGateway createPaymentGateway(String gatewayType){
        if (gatewayType.equals("PayPal")){
            return new PayPalGateway();
        }else{
            return new StripeGateway();
        }
    }
    @Override
    public Invoice createInvoice(){
        return new USInvoice();
    } 
}


