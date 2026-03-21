package src.creational.AbstractFactory;

public class IndiaFactory implements RegionFactory{
    
    @Override
    public PaymentGateway createPaymentGateway(String gatewayType){
        if (gatewayType.equals("RazorPay")){
            return new RazorPayGateway();
        }else{
            return new PayUGateway();
        }
    }

    @Override
    public Invoice createInvoice(){
        return new GSTInvoice();
    }
}
