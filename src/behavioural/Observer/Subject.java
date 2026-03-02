package src.behavioural.Observer;

// This is the contract that every Subject should have
interface Subject {
    // method to add subscriber
    void subscribe(Subscriber subscriber);
    // unsubscribe them
    void unsubscribe(Subscriber subscriber);
    // notify them
    void notifySubscriber(String title);
    
}
