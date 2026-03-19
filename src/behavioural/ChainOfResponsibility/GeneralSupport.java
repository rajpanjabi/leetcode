package src.behavioural.ChainOfResponsibility;

public class GeneralSupport extends SupportHandler{
    @Override
    public void handleRequest(String request){
        if (request.equalsIgnoreCase("general")){
        System.out.println("General Support is Handling request");
        }
        else if (nextHandler!=null){
            nextHandler.handleRequest(request);
        }
    }
    
    
}


