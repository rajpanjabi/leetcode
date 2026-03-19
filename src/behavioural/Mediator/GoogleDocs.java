package behavioural.Mediator;

import java.util.ArrayList;

public class GoogleDocs implements DocumentMediator{
    // this is the concrete doc object
    ArrayList<User> users = new ArrayList<>();
    @Override
    public void broadcast(String change, User user){
        for (User u :users){
            if (u != user){
                System.out.println(u.getName()+ " recieved change: "+ change +" from User: "+ user.getName());
            }
        }
        
    }

    @Override
    public void join(User user){
        users.add(user);
    }

}
