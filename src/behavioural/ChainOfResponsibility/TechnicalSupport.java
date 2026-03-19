package src.behavioural.ChainOfResponsibility;

public class TechnicalSupport extends SupportHandler{
    @Override
    public void handleRequest(String request){
        if (request.equalsIgnoreCase("technical")){
            System.out.println("Technical Support handling request");
        }
        else if (nextHandler!=null){
            nextHandler.handleRequest(request);
        }
    }
}

