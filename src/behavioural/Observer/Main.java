package src.behavioural.Observer;

public class Main {

    public static void main(String[] args) {
        YoutubeSubject takeuforward = new YoutubeSubject("TakeUForward");
        Subscriber user1 = new EmailSubscriber("user1@gmail.com");
        Subscriber user2 = new MobileSubscriber("user2");
        takeuforward.subscribe(user1);
        takeuforward.subscribe(user2);
        takeuforward.uploadVideo("LLD");
        takeuforward.uploadVideo("DSA");
    }
    
}
