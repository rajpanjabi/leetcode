package src.behavioural.ChainOfResponsibility;

public class DeliverySupport extends SupportHandler {
    @Override
    public void handleRequest(String request){
        if (request.equalsIgnoreCase("delivery")){
            System.out.println("Delivery Support handling request");
        }
        else if (nextHandler!=null){
            nextHandler.handleRequest(request);
        }
        else{
            System.out.println("Manual Intervention required");
        }
    }
}
