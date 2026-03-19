package src.behavioural.ChainOfResponsibility;

public class Main {
    
    public static void main (String[] args){
         // Client side code
    
    // First we create handlers
    SupportHandler general = new GeneralSupport();
    SupportHandler billing = new BillingSupport();
    SupportHandler delivery = new DeliverySupport();
    SupportHandler technical = new TechnicalSupport();
    general.setNextHandler(billing);
    billing.setNextHandler(technical);
    technical.setNextHandler(delivery);

    // Now simulate request
    general.handleRequest("billing");
    


    }
   
}
