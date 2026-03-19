package behavioural.Mediator;

public class User {
    private String name;
    private DocumentMediator mediator;

    public User(String name, DocumentMediator mediator){
        this.name=name;
        this.mediator=mediator;
    }
    public String getName(){
        return this.name;
    }
    
    public void makeChange(String change){
        mediator.broadcast(change, this);
    }
    
}
