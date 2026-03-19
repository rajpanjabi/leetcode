package src.behavioural.ChainOfResponsibility;

public class BillingSupport extends SupportHandler {
    @Override
    public void handleRequest(String request){
        if (request.equalsIgnoreCase("billing")){
            System.out.println("Billing Support handling request");
        }
        else if (nextHandler!=null){
            nextHandler.handleRequest(request);
        }
    }
}

