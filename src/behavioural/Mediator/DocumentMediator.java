package behavioural.Mediator;

public interface DocumentMediator {
    void broadcast(String change, User user);
    void join(User user);
}
