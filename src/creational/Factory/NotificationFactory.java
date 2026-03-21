package src.creational.Factory;

public class NotificationFactory {
    public static Notification getNotification(String type) {
        if (type.equalsIgnoreCase("EMAIL")) {
            return new EmailNotification();
        } else if (type.equalsIgnoreCase("SMS")) {
            return new SMSNotification();
        } else{
        throw new IllegalArgumentException("Unknown notification type: " + type);
        }
    }
}
