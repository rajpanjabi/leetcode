package src.behavioural.Observer;

import java.util.ArrayList;
import java.util.List;

public class YoutubeSubject implements Subject{
    // should have name of Channel
    private String name;
    private List<Subscriber> subscribers= new ArrayList<>();
    public YoutubeSubject(String name){
        this.name=name;
    }

    @Override
    public void subscribe(Subscriber subscriber){
        subscribers.add(subscriber);
    }
    @Override
    public void unsubscribe(Subscriber subscriber){
        subscribers.remove(subscriber);
    }
    @Override
    public void notifySubscriber(String title){
        for (Subscriber s : subscribers){
            s.update(title);
        }
    }
    public void uploadVideo(String title){
        System.out.println("Uploading video: "+ title +" in "+name);
        notifySubscriber(title);
    }
    
}
