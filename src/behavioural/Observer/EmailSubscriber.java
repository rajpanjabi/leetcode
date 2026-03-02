package src.behavioural.Observer;

public class EmailSubscriber implements Subscriber{
    private String email;
    public EmailSubscriber(String email){
        this.email=email;
    }
    @Override
    public void update(String title){
        System.out.println("Sending email notification to "+email+ " for new video: "+title);
    }
}

