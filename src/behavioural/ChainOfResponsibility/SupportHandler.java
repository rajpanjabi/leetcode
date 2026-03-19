package src.behavioural.ChainOfResponsibility;


public abstract class SupportHandler{
    // this is the base abstract class with reference to the next handler, logic for current handler
    protected SupportHandler nextHandler;

    public abstract void handleRequest(String request);

    public void setNextHandler(SupportHandler nextHandler){
        this.nextHandler=nextHandler;
    }

}