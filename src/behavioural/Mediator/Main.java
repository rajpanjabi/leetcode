package behavioural.Mediator;

public class Main {


    public static void main(String[] args) {
        // First create the collab document
        GoogleDocs document = new GoogleDocs();


        // create users with the document
        User Bob = new User("Bob",document);
        User Alice = new User("Alice",document);
        User Charlie = new User("Charlie",document);

        // we also need to add the users to the document session
        document.join(Bob);
        document.join(Alice);
        document.join(Charlie);

        // Now lets make change and see if the other users are notified
        // Here Alice make change to the collab doc so other users need to be notified
        Alice.makeChange("Adding AI/ML Config to the class");

         // Here Alice make change to the collab doc so other users need to be notified
        Charlie.makeChange("Created new Dashboard for logs and metrics");


        // Now 
    }
    
}
